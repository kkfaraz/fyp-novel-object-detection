import logging
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from backend.services.registry.model_registry import ModelRegistry
from backend.services.strategies.base import DatasetStrategy
import torchvision.transforms.functional as TF
import open_clip

logger = logging.getLogger("CLIPSemanticVerifier")

class CLIPSemanticVerifier:
    def __init__(self, device: torch.device):
        self.device = device
        self.registry = ModelRegistry()
        self.clip = self.registry.load_clip(device)
        self.preprocess = self.registry.get_model("clip_preprocess")
        self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP') if hasattr(open_clip, 'get_tokenizer') else None
        
        self.templates = [
            "a photo of a {}.",
            "this is a photo of a {}.",
            "a cropped photo of a {}.",
            "a close-up photo of a {}.",
            "a centered photo of a {}.",
            "a clear photo of a {}.",
            "an image of a {}.",
            "a picture of a {}."
        ]

    def _get_vild_templates(self) -> List[str]:
        return self.templates

    def crop_regions(self, image: Image.Image, boxes: torch.Tensor, pad_scale: float = 1.1) -> Tuple[List[Image.Image], List[int]]:
        """
        Crops regions from the image based on bounding boxes.
        Adds padding context for smaller objects.
        """
        crops = []
        valid_indices = []
        img_w, img_h = image.size

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box.tolist())
            w = x2 - x1
            h = y2 - y1
            
            if w < 10 or h < 10:
                continue

            cx, cy = x1 + w / 2, y1 + h / 2
            
            # Apply padding scale
            new_w = w * pad_scale
            new_h = h * pad_scale
            
            px1 = max(0, int(cx - new_w / 2))
            py1 = max(0, int(cy - new_h / 2))
            px2 = min(img_w, int(cx + new_w / 2))
            py2 = min(img_h, int(cy + new_h / 2))
            
            if (px2 - px1) < 8 or (py2 - py1) < 8:
                continue
                
            crop = image.crop((px1, py1, px2, py2)).convert("RGB")
            crops.append(crop)
            valid_indices.append(idx)

        return crops, valid_indices

    @torch.no_grad()
    def encode_text(self, class_names: List[str]) -> torch.Tensor:
        """
        Encodes class names using multiple templates and averages the embeddings.
        """
        templates = self._get_vild_templates()
        target_device = self.device
        target_dtype = next(self.clip.parameters()).dtype

        class_features = []
        for name in class_names:
            prompts = [temp.format(name) for temp in templates]
            if self.tokenizer:
                tokens = self.tokenizer(prompts).to(target_device)
            else:
                tokens = open_clip.tokenize(prompts).to(target_device)
                
            feats = self.clip.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)
            mean_feat = feats.mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=-1)
            class_features.append(mean_feat)

        return torch.stack(class_features, dim=0)

    @torch.no_grad()
    def encode_images(self, crops: List[Image.Image]) -> torch.Tensor:
        """
        Encodes crop images using multi-scale crop aggregation and sub-batching.
        """
        if not crops:
            embed_dim = getattr(self.clip, 'output_dim', 768)
            return torch.empty((0, embed_dim), device=self.device)

        target_dtype = next(self.clip.parameters()).dtype
        scales = [1.0, 1.2]
        all_scales_features = []

        for scale in scales:
            scaled_images = []
            for img in crops:
                if scale == 1.0:
                    scaled_images.append(img)
                else:
                    w, h = img.size
                    pw, ph = int(w * (scale - 1) / 2), int(h * (scale - 1) / 2)
                    try:
                        padded = TF.pad(img, (pw, ph, pw, ph), padding_mode='reflect')
                    except Exception:
                        padded = TF.pad(img, (pw, ph, pw, ph), padding_mode='edge')
                    scaled_images.append(padded)

            # Process in small chunks to avoid VRAM overflow
            chunk_size = 4
            scale_feats = []
            for i in range(0, len(scaled_images), chunk_size):
                chunk = scaled_images[i : i + chunk_size]
                tensors = torch.stack([self.preprocess(c) for c in chunk]).to(self.device, dtype=target_dtype)
                feats = self.clip.encode_image(tensors)
                scale_feats.append(feats)
            
            all_scales_features.append(torch.cat(scale_feats, dim=0))

        stacked = torch.stack(all_scales_features, dim=0)
        mean_feats = stacked.mean(dim=0)
        return F.normalize(mean_feats, dim=-1)

    def verify(
        self,
        image_path: str,
        boxes: torch.Tensor,
        strategy: DatasetStrategy,
        clip_threshold: float = 0.25
    ) -> Dict[str, torch.Tensor]:
        """
        Verifies and refines proposal categories using CLIP.
        """
        if len(boxes) == 0:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device),
                "labels": torch.empty((0,), dtype=torch.long, device=self.device)
            }

        image = Image.open(image_path).convert("RGB")
        crops, valid_indices = self.crop_regions(image, boxes)
        
        if not crops:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device),
                "labels": torch.empty((0,), dtype=torch.long, device=self.device)
            }

        image_features = self.encode_images(crops)
        text_features = self.encode_text(strategy.get_classes())

        # Compute cosine similarity
        similarities = image_features @ text_features.T
        scores, labels = similarities.max(dim=1)

        keep_mask = scores > clip_threshold
        keep_indices = torch.nonzero(keep_mask).squeeze(1).tolist()

        if not keep_indices:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device),
                "labels": torch.empty((0,), dtype=torch.long, device=self.device)
            }

        orig_keep_indices = [valid_indices[ki] for ki in keep_indices]
        
        return {
            "boxes": boxes[orig_keep_indices],
            "scores": scores[keep_indices],
            "labels": labels[keep_indices]
        }
