import logging
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from backend.services.detectors.base import BaseDetector
from backend.services.registry.model_registry import ModelRegistry
from backend.services.strategies.base import DatasetStrategy
from backend.services.preprocessing.transforms import Compose, RandomResize, ToTensor, Normalize
from PIL import Image
import numpy as np

logger = logging.getLogger("GroundingDINODetector")

class GroundingDINODetector(BaseDetector):
    def __init__(self, device: torch.device):
        self.device = device
        self.registry = ModelRegistry()
        self.gdino = self.registry.load_gdino(device)
        self.tokenizer = self.gdino.tokenizer

        # Image transform for GDINO
        self.transform = Compose([
            RandomResize([800], max_size=1333),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, file_name: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Preprocesses image from path for Grounding DINO."""
        image_src = Image.open(file_name).convert("RGB")
        w, h = image_src.size
        image_transformed, _ = self.transform(image_src, None)
        return image_transformed[None].to(self.device), (h, w)

    def detect(self, inputs: List[Dict[str, Any]], strategy: DatasetStrategy, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Runs Grounding DINO inference using the dataset vocabulary defined by the strategy.
        """
        file_name = inputs[0]["file_name"]
        gdino_image, (h, w) = self.preprocess(file_name)

        # Get text prompts and positive maps from the strategy
        class_len_per_prompt = kwargs.get("class_len_per_prompt", 81)
        text_prompt_list, positive_map_list = strategy.get_prompts(self.tokenizer, class_len_per_prompt)
        
        all_prob_to_token = []
        all_out_bbox = []
        
        GDINO_BATCH_SIZE = 1
        with torch.no_grad(), torch.cuda.amp.autocast():
            for i in range(0, len(text_prompt_list), GDINO_BATCH_SIZE):
                batch_captions = text_prompt_list[i:i+GDINO_BATCH_SIZE]
                batched_image = gdino_image.repeat(len(batch_captions), 1, 1, 1)
                curr_output = self.gdino(batched_image, captions=batch_captions)
                all_out_bbox.append(curr_output["pred_boxes"])
                all_prob_to_token.append(curr_output["pred_logits"].sigmoid())

        if not all_prob_to_token:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device),
                "labels": torch.empty((0,), dtype=torch.long, device=self.device)
            }

        prob_to_token = torch.cat(all_prob_to_token, dim=0)
        out_bbox = torch.cat(all_out_bbox, dim=0)

        prob_to_label_list = []
        for i in range(prob_to_token.shape[0]):
            curr_prob_to_label = prob_to_token[i] @ positive_map_list[i].to(prob_to_token.device).T
            prob_to_label_list.append(curr_prob_to_label)

        prob_to_label = torch.cat(prob_to_label_list, dim=1)
        
        thresholds = strategy.get_thresholds()
        box_threshold = kwargs.get("box_threshold", thresholds.get("box_threshold", 0.35))
        
        # Select top predictions
        topk_limit = min(900, prob_to_label.view(-1).shape[0])
        topk_values, topk_idxs = torch.topk(prob_to_label.view(-1), topk_limit, 0)
        
        labels = topk_idxs % prob_to_label.shape[1]
        topk_boxes_idx = topk_idxs // prob_to_label.shape[1]
        
        # Filter predictions based on confidence threshold
        keep_mask = topk_values > box_threshold
        gdino_scores = topk_values[keep_mask]
        labels = labels[keep_mask]
        topk_boxes_idx = topk_boxes_idx[keep_mask]
        
        if len(gdino_scores) == 0:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device),
                "labels": torch.empty((0,), dtype=torch.long, device=self.device)
            }

        # Convert boxes from cxcywh (0-1 scaled) to xyxy (absolute coordinates)
        raw_boxes = out_bbox.view(-1, 4)[topk_boxes_idx]
        
        # Scale back to original resolution
        cxcywh_boxes = raw_boxes * torch.tensor([w, h, w, h], dtype=torch.float, device=self.device)
        
        # Convert cxcywh to xyxy
        x1y1x2y2_boxes = torch.zeros_like(cxcywh_boxes)
        x1y1x2y2_boxes[:, 0] = cxcywh_boxes[:, 0] - cxcywh_boxes[:, 2] / 2
        x1y1x2y2_boxes[:, 1] = cxcywh_boxes[:, 1] - cxcywh_boxes[:, 3] / 2
        x1y1x2y2_boxes[:, 2] = cxcywh_boxes[:, 0] + cxcywh_boxes[:, 2] / 2
        x1y1x2y2_boxes[:, 3] = cxcywh_boxes[:, 1] + cxcywh_boxes[:, 3] / 2
        
        # Clip boxes to image boundaries
        x1y1x2y2_boxes[:, 0] = x1y1x2y2_boxes[:, 0].clamp(0, w)
        x1y1x2y2_boxes[:, 1] = x1y1x2y2_boxes[:, 1].clamp(0, h)
        x1y1x2y2_boxes[:, 2] = x1y1x2y2_boxes[:, 2].clamp(0, w)
        x1y1x2y2_boxes[:, 3] = x1y1x2y2_boxes[:, 3].clamp(0, h)

        return {
            "boxes": x1y1x2y2_boxes,
            "scores": gdino_scores,
            "labels": labels
        }
