"""
Stage-2 Semantic Reasoning: VLRM + CLIP Fusion
================================================
 
Replaces mock objects with real pretrained models:
- VLRM: Fine-tuned BLIP2 (sashakunitsyn/vlrm-blip2-opt-2.7b) for region captioning
- CLIP: open_clip ViT-B-32 for image-text similarity scoring
 
Pipeline per ROI:
  1. Crop ROI from image (skip if < 10×10 px)
  2. VLRM generates a caption for the crop
  3. CLIP text-encodes the caption → cosine similarity to class name embeddings
  4. CLIP image-encodes the crop → cosine similarity to class name embeddings
  5. Fuse: α × caption_similarity + (1-α) × image_similarity
 
Output: [N, K] semantic confidence matrix (N ROIs × K classes)
"""
 
import torch
import torch.nn.functional as F
import numpy as np
import logging
import gc
import hashlib
import pickle
import os
import time
from datetime import datetime
from PIL import Image
from typing import List, Optional, Tuple, Any, Dict
 
# Suppress transformers noise
logging.getLogger("transformers").setLevel(logging.ERROR)

# Internal configuration (not exposed via CLI)
# LPC is now controlled per-instance via Stage2_VLRM_CLIP_Fusion(use_lpc=...)

from vild_templates import get_vild_templates


# ==============================================================================
# VLRM Caption Generator (Real BLIP2)
# ==============================================================================
 
class VLRMCaptioner:
    """
    VLRM-tuned BLIP2 for region-level caption generation.
    
    Loaded on CPU by default. Call prepare() to move to GPU, finish() to offload.
    """
    
    def __init__(self, model_name="sashakunitsyn/vlrm-blip2-opt-2.7b", device="cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._on_gpu = False
        self._load_model()
    
    def _load_model(self):
        """Load VLRM-tuned BLIP2 onto CPU."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration
        
        try:
            print(f"[VLRM] Loading {self.model_name} on CPU...")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.float16
            )
            self.model.eval()
            print("[VLRM] ✓ Loaded on CPU (will move to GPU on demand)")
        except Exception as e:
            print(f"[VLRM ERROR] Failed to load model: {str(e)}")
            raise e
    
    def prepare(self):
        """Move model to GPU for inference with OOM safety."""
        if not self._on_gpu and self.model is not None:
            if self.device == "cpu":
                self.model = self.model.to("cpu").float()
                self._on_gpu = False
                return
            try:
                print(f"[VLRM] Attempting to move to {self.device}...")
                self.model = self.model.to(self.device).half()
                self._on_gpu = True
                if torch.cuda.is_available():
                    free_mem, _ = torch.cuda.mem_get_info()
                    print(f"[VLRM] ✓ Moved to GPU (FP16, {free_mem/1e9:.1f}GB free)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("[VLRM WARNING] Failed to load on GPU. Staying on CPU.")
                    self.model = self.model.to("cpu").float()
                    self._on_gpu = False
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else: 
                    print(f"[VLRM ERROR] Runtime error during prepare: {str(e)}")
                    raise e
    
    def finish(self, force_offload=False):
        """Offload model back to CPU to free GPU memory only if forced."""
        try:
            if force_offload and self._on_gpu and self.model is not None:
                self.model = self.model.to("cpu")
                self._on_gpu = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as e:
            print(f"[VLRM ERROR] Error during finish (offload): {str(e)}")
    
    @torch.no_grad()
    def generate_caption(self, pil_image: Image.Image) -> str:
        """Generate a caption for a single PIL image crop."""
        if self.model is None:
            return "an object"
        
        # Ensure RGB and minimum size
        pil_image = pil_image.convert("RGB")
        if pil_image.width < 8 or pil_image.height < 8:
            pil_image = pil_image.resize(
                (max(pil_image.width, 8), max(pil_image.height, 8)), Image.BICUBIC
            )
        
        try:
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            inputs = self.processor(images=pil_image, return_tensors="pt").to(
                model_device, dtype=model_dtype
            )
            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
            )
            
            caption = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            
            return caption if caption else "an object"
            
        except Exception as e:
            print(f"[VLRM ERROR] Caption generation failed: {str(e)}")
            return "an object"

    @torch.no_grad()
    def generate_captions_batch(self, pil_images: List[Image.Image]) -> List[str]:
        """Generate captions for a batch of PIL image crops."""
        if self.model is None or not pil_images:
            return ["an object"] * len(pil_images)
        
        # Preprocess images (convert to RGB, resize if too small)
        processed_images = []
        for img in pil_images:
            img = img.convert("RGB")
            if img.width < 8 or img.height < 8:
                img = img.resize((max(img.width, 8), max(img.height, 8)), Image.BICUBIC)
            processed_images.append(img)
            
        try:
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype
            
            # Batch process images
            inputs = self.processor(images=processed_images, return_tensors="pt").to(
                model_device, dtype=model_dtype
            )
            
            # Batch generate captions
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=1,
                do_sample=False,
            )
            
            captions = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return [c.strip() if c.strip() else "an object" for c in captions]
            
        except Exception as e:
            print(f"[VLRM WARNING] Batch caption generation failed: {str(e)}. Cleaning cache and falling back to sequential.")
            # Clear GPU cache before falling back
            torch.cuda.empty_cache()
            gc.collect()
            
            # Fallback to sequential generation with inner try-except
            captions = []
            for img in pil_images:
                try:
                    cap = self.generate_caption(img)
                    captions.append(cap)
                except Exception as inner_e:
                    print(f"[VLRM ERROR] Sequential fallback failed for crop: {str(inner_e)}. Clearing cache and returning fallback text.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    captions.append("an object")
            return captions
 
 
# ==============================================================================
# CLIP Encoder (Real open_clip)
# ==============================================================================
 
class CLIPEncoder:
    """
    CLIP encoder for both image and text embedding + similarity.
    
    Uses open_clip ViT-B-32 (always on GPU, ~0.6GB).
    """
    
    def __init__(self, model_name="ViT-SO400M-14-SigLIP", pretrained="webli", device="cuda"):
        import open_clip
        import gc
        
        print(f"[SigLIP] Loading {model_name} ({pretrained})...")
        self.device = device
        
        try:
            # Try loading on GPU first
            if device == "cuda":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            
            if device == "cuda":
                # Convert to half to save memory
                self.model = self.model.half()
                print(f"[SigLIP] ✓ Loaded on {device} (FP16)")
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device == "cuda":
                print(f"[SigLIP WARNING] Failed to load on GPU. Falling back to CPU.")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device="cpu"
                )
                self.tokenizer = open_clip.get_tokenizer(model_name)
                self.model.eval()
                self.device = "cpu"
                print(f"[SigLIP] ✓ Loaded on CPU")
            else:
                print(f"[SigLIP ERROR] Failed to initialize SigLIP: {str(e)}")
                raise e
    
    @torch.no_grad()
    def encode_class_names_with_templates(self, class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode class names into SigLIP text embeddings using ViLD templates.
        
        Returns:
            mean_features: (K, D) normalized mean embeddings for each class
            covariance_inv: (D, D) inverse of the shared covariance matrix
        """
        templates = get_vild_templates()

        all_features = []
        batch_size = 1000
        
        try:
            target_device = self.device
            target_dtype = next(self.model.parameters()).dtype

            for c in class_names:
                prompts = [t.format(c) for t in templates]
                class_embeddings_list = []
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i+batch_size]
                    tokens = self.tokenizer(batch_prompts).to(target_device)
                    with torch.no_grad():
                        feats = self.model.encode_text(tokens)
                        feats = F.normalize(feats, dim=-1) # (B, D)
                    class_embeddings_list.append(feats.float().cpu())
                
                class_embeddings = torch.cat(class_embeddings_list, dim=0) # (T, D) on CPU
                all_features.append(class_embeddings)
                
            all_features_stack = torch.stack(all_features, dim=0) # (K, T, D) on CPU
            K, T, D = all_features_stack.shape
            
            mean_features = all_features_stack.mean(dim=1) # (K, D) on CPU
            mean_features = F.normalize(mean_features, dim=-1)
            
            centered_features = (all_features_stack - mean_features.unsqueeze(1)).reshape(-1, D)
            cov_emp = torch.matmul(centered_features.T, centered_features) / (K * T - 1)

            # Tikhonov regularization λ=0.1 (matches official WACV 2025 implementation).
            # Ledoit-Wolf λ=0.5 was too aggressive, collapsing class-discriminative structure.
            lambda_reg = 0.1
            cov_shrink = cov_emp + lambda_reg * torch.eye(D, dtype=cov_emp.dtype)
            cov_inv = torch.linalg.inv(cov_shrink.float()).to(dtype=cov_shrink.dtype)

            print(f"[CLIP] Computed Tikhonov-regularized covariance (λ={lambda_reg}) from {T} templates/class on CPU. Det(Cov): {torch.det(cov_shrink.float()):.2e}")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return mean_features.to(device=target_device, dtype=target_dtype), cov_inv.to(device=target_device, dtype=target_dtype)
        except Exception as e:
            print(f"[SigLIP ERROR] Class encoding failed: {str(e)}")
            raise e
 
    @torch.no_grad()
    def compute_mahalanobis(self, features_x: torch.Tensor, mean_features: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
        """Compute negative Mahalanobis distance."""
        try:
            dtype = features_x.dtype
            features_x_f = features_x.float()
            mean_features_f = mean_features.float()
            cov_inv_f = cov_inv.float()

            N, D = features_x_f.shape
            x_Sinv = features_x_f @ cov_inv_f
            term1 = (x_Sinv * features_x_f).sum(dim=1) # (N,)
            term2 = -2 * (x_Sinv @ mean_features_f.T) # (N, K)
            u_Sinv = mean_features_f @ cov_inv_f # (K, D)
            term3 = (u_Sinv * mean_features_f).sum(dim=1) # (K,)
            dist = term1.unsqueeze(1) + term2 + term3.unsqueeze(0) # (N, K)
            return (-dist).to(dtype=dtype)
        except Exception as e:
            print(f"[CLIP ERROR] Mahalanobis computation failed: {str(e)}")
            # Return zero similarity as fallback
            return torch.zeros((features_x.shape[0], mean_features.shape[0]), device=features_x.device, dtype=features_x.dtype)
 
    @torch.no_grad()
    def encode_class_names(self, class_names: List[str]) -> torch.Tensor:
        try:
            prompts = [f"a photo of a {c}" for c in class_names]
            tokens = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
            return text_features
        except Exception as e:
            print(f"[CLIP ERROR] Basic class encoding failed: {str(e)}")
            raise e
    
    @torch.no_grad()
    def encode_images_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        embed_dim = getattr(self.model, 'output_dim', 768)
        if len(pil_images) == 0:
            return torch.empty(0, embed_dim, device=self.device)
            
        try:
            dtype = next(self.model.parameters()).dtype
            
            # Mod 2: Multi-Scale Crop Aggregation [1.0x, 1.2x]
            # Only 2 scales: 1.4x removed — adds too much context and blurs small/rare objects.
            # 'reflect' padding avoids edge-duplicate artifacts that 'edge' padding creates
            # at bounding box borders, which confuses fine-grained LVIS categories.
            import torchvision.transforms.functional as TF
            
            all_scales_features = []
            scales = [1.0, 1.2]  # Reduced from [1.0, 1.2, 1.4]
            
            for scale in scales:
                scaled_images = []
                for img in pil_images:
                    if scale == 1.0:
                        scaled_images.append(img)
                    else:
                        w, h = img.size
                        pad_w, pad_h = int(w * (scale - 1) / 2), int(h * (scale - 1) / 2)
                        # Use reflect padding to avoid border artifact duplication
                        try:
                            padded = TF.pad(img, (pad_w, pad_h, pad_w, pad_h), padding_mode='reflect')
                        except Exception:
                            # reflect requires image larger than pad; fall back to edge
                            padded = TF.pad(img, (pad_w, pad_h, pad_w, pad_h), padding_mode='edge')
                        scaled_images.append(padded)
                        
                # Sub-batched image encoding to keep VRAM usage low and avoid CUDA OOM
                sub_batch_size = 4
                scale_feats = []
                for chunk_idx in range(0, len(scaled_images), sub_batch_size):
                    chunk = scaled_images[chunk_idx : chunk_idx + sub_batch_size]
                    tensors = torch.stack([self.preprocess(img) for img in chunk]).to(self.device, dtype=dtype)
                    feats = self.model.encode_image(tensors)
                    scale_feats.append(feats)
                feats = torch.cat(scale_feats, dim=0)
                all_scales_features.append(feats)
                
            # Aggregate and normalize
            stacked_feats = torch.stack(all_scales_features, dim=0) # (3, N, D)
            agg_features = stacked_feats.mean(dim=0)
            agg_features = F.normalize(agg_features, dim=-1)
            
            return agg_features
        except Exception as e:
            print(f"[CLIP ERROR] Image batch encoding failed: {str(e)}")
            return torch.empty(0, 512, device=self.device)
    
    @torch.no_grad()
    def encode_captions_batch(self, captions: List[str]) -> torch.Tensor:
        embed_dim = getattr(self.model, 'output_dim', 768)
        if len(captions) == 0:
            return torch.empty(0, embed_dim, device=self.device)
        try:
            tokens = self.tokenizer(captions).to(self.device)
            caption_features = self.model.encode_text(tokens)
            caption_features = F.normalize(caption_features, dim=-1)
            return caption_features
        except Exception as e:
            print(f"[CLIP ERROR] Caption batch encoding failed: {str(e)}")
            return torch.empty(0, 512, device=self.device)
 
 
# ==============================================================================
# Low-Density Prior Constraint (LPC)
# ==============================================================================
 
class LowDensityPriorConstraint:
    """
    Low-Density Prior Constraint (LPC) for Known/Unknown separation.
    Reference: OOVDet (Su et al., 2026) — Section 3.4
    """
    def __init__(self, bandwidth: float = 0.8, density_threshold: float = None):
        self.bandwidth = bandwidth
        self.density_threshold = density_threshold
    
    @torch.no_grad()
    def estimate_log_density(self, query_features: torch.Tensor, known_feature_bank: torch.Tensor) -> torch.Tensor:
        """
        Estimate density of query features in the space of known features.
        Higher log_density means the point is closer to known clusters.
        """
        N, D = query_features.shape
        M, _ = known_feature_bank.shape
        h = self.bandwidth
        
        # Cosine similarity to distance: d^2 = 2(1-cos)
        cosine_sim = query_features @ known_feature_bank.T
        sq_dist = 2.0 * (1.0 - cosine_sim)
        
        # Gaussian kernel: K(u) = exp(-u^2 / 2h^2)
        log_kernels = -sq_dist / (2.0 * h * h)
        log_densities = torch.logsumexp(log_kernels, dim=1) - np.log(M)
        return log_densities
    
    @torch.no_grad()
    def apply(self, semantic_scores: torch.Tensor, query_features: torch.Tensor, known_feature_bank: torch.Tensor) -> Dict[str, torch.Tensor]:
        log_densities = self.estimate_log_density(query_features, known_feature_bank)
        
        if self.density_threshold is not None:
            threshold = self.density_threshold
        else:
            threshold = log_densities.median().item()
        
        # Boost low-density regions: weight is high when density is low
        scale = 5.0
        # Sigmoid(scale * (threshold - log_density)) -> 1.0 when density << threshold
        lpc_weights = torch.sigmoid(scale * (threshold - log_densities))
        
        adjusted_scores = semantic_scores * lpc_weights.unsqueeze(1)
        
        return {
            "original_scores": semantic_scores,
            "lpc_adjusted_scores": adjusted_scores,
            "lpc_weights": lpc_weights,
            "unknown_flags": (lpc_weights > 0.5),
            "density_values": log_densities
        }
 
 
# ==============================================================================
# Stage-2 Processor: VLRM + CLIP Fusion
# ==============================================================================
 
class Stage2_VLRM_CLIP_Fusion:
    def __init__(
        self,
        device="cuda",
        fusion_alpha=0.4,
        vlrm_model_name="sashakunitsyn/vlrm-blip2-opt-2.7b",
        clip_model_name="ViT-SO400M-14-SigLIP",
        clip_pretrained="webli",
        cache_dir="cache/vlrm_outputs",
        use_lpc=True,
        use_saeg=False,
    ):
        self.device = device
        self.fusion_alpha = fusion_alpha
        self.cache_dir = cache_dir # Should be cache/vlrm_outputs_lvis
        self.use_lpc = use_lpc
        self.use_saeg = use_saeg
        
        print("=" * 60)
        print("Initializing Stage-2: VLRM + CLIP Fusion")
        print("=" * 60)
        self.vlrm = VLRMCaptioner(model_name=vlrm_model_name, device=device)
        self.clip = CLIPEncoder(model_name=clip_model_name, pretrained=clip_pretrained, device=device)
        
        self._class_mean_features = None
        self._class_cov_inv = None
        self._cached_class_names = None
        
        # Load LVIS frequencies for Mod 5 and Mod 7
        try:
            from novel_object_detection.lvis_calibration import get_lvis_frequency_groups
            self.freq_groups, self.class_counts = get_lvis_frequency_groups("lvis_v1_val")
            self.class_counts = self.class_counts.to(device)
            # Gamma factor for Frequency-Aware Mahalanobis (Mod 5)
            self.gamma_c = 1.0 / torch.log(1.1 + self.class_counts)
            
            # Temperatures for Class-Specific Temperature Scaling (Mod 7)
            self.temperatures = torch.ones(1203, device=device) * 0.07 # Default common
            for idx in self.freq_groups.get("rare", []):
                if idx < 1203: self.temperatures[idx] = 0.05
            for idx in self.freq_groups.get("frequent", []):
                if idx < 1203: self.temperatures[idx] = 0.10
        except Exception as e:
            print(f"[Stage-2 WARNING] Could not load frequency groups: {e}")
            self.gamma_c = torch.ones(1203, device=device)
            self.temperatures = torch.ones(1203, device=device) * 0.07
            
        print(f"[Stage-2] ✓ Ready | fusion_alpha={fusion_alpha} | LPC={'ON' if use_lpc else 'OFF'} | SAEG={'ON' if use_saeg else 'OFF'}")
        print(f"[Stage-2]   VLRM: {vlrm_model_name} (CPU → GPU on demand)")
        print(f"[Stage-2]   SigLIP: {clip_model_name}/{clip_pretrained} (GPU)")
        print("=" * 60)
 
    def _get_class_stats(self, class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._class_mean_features is None or self._cached_class_names != class_names:
            if self.use_saeg:
                print("[Stage-2] Computing SAEG statistics (Synonym Averaged Embedding Generator)...")
                self._class_mean_features, self._class_cov_inv = self._get_class_stats_saeg(class_names)
            else:
                print("[Stage-2] Computing Mahalanobis statistics (this may take a moment)...")
                self._class_mean_features, self._class_cov_inv = self.clip.encode_class_names_with_templates(class_names)
            self._cached_class_names = class_names
        return self._class_mean_features, self._class_cov_inv

    def _get_class_stats_saeg(self, class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        from saeg_module import SAEGTextFeatureGenerator, load_class_to_synonyms
        from vild_templates import get_vild_templates

        is_voc = len(class_names) == 20 or any(c in ["aeroplane", "tvmonitor", "pottedplant"] for c in class_names)
        if is_voc:
            try:
                from datasets.voc.voc_class_utils import get_voc_synonyms
                synonyms = get_voc_synonyms()
                print("[SAEG] Detected PASCAL VOC class list. Loaded VOC class synonyms successfully.")
            except Exception as e:
                print(f"[SAEG Warning] Could not load VOC synonyms: {e}. Falling back to default.")
                synonyms = load_class_to_synonyms("lvis_original_class_to_synonyms.pkl")
        else:
            synonyms = load_class_to_synonyms("lvis_original_class_to_synonyms.pkl")

        templates = get_vild_templates()

        saeg = SAEGTextFeatureGenerator(
            clip_model=self.clip.model,
            tokenizer=self.clip.tokenizer,
            templates=templates,
            class_to_synonyms=synonyms,
            device=self.device,
            show_progress=True
        )
        saeg_features = saeg.generate_text_features(class_names)

        # Compute covariance from template embeddings for Mahalanobis distance
        _, cov_inv = self.clip.encode_class_names_with_templates(class_names)

        print(f"[SAEG] Features: {saeg_features.shape}, Cov: {cov_inv.shape}")
        return saeg_features, cov_inv
 
    def _compute_input_hash(self, image_path: str, config: dict) -> str:
        """
        Stable cache hash keyed only on image filename + config.
        Does NOT include ROI coordinates — those changed between formats
        and would invalidate all existing cache files.
        """
        hasher = hashlib.sha256()
        hasher.update(os.path.basename(image_path).encode())
        config_str = str(sorted(config.items()))
        hasher.update(config_str.encode())
        return hasher.hexdigest()

    def _match_cached_indices(
        self,
        cached_regions: np.ndarray,
        cached_valid_indices: List[int],
        current_boxes: torch.Tensor,
        img_size: tuple,
    ) -> List[tuple]:
        if cached_regions is None or len(cached_regions) == 0 or len(current_boxes) == 0:
            return []

        w, h = img_size
        cached_t = torch.as_tensor(cached_regions, dtype=torch.float32)
        current_t = current_boxes.float().cpu()

        pairs = []
        for ci, vi in enumerate(cached_valid_indices):
            if vi >= len(current_boxes):
                continue

            cx1 = (cached_t[ci, 0] + cached_t[ci, 2]) / (2.0 * w)
            cy1 = (cached_t[ci, 1] + cached_t[ci, 3]) / (2.0 * h)
            aw1 = (cached_t[ci, 2] - cached_t[ci, 0]) / w
            ah1 = (cached_t[ci, 3] - cached_t[ci, 1]) / h

            box = current_t[vi]
            cx2 = (box[0] + box[2]) / (2.0 * w)
            cy2 = (box[1] + box[3]) / (2.0 * h)
            aw2 = (box[2] - box[0]) / w
            ah2 = (box[3] - box[1]) / h

            center_dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            area_ratio = (aw2 * ah2) / max(aw1 * ah1, 1e-8)

            if center_dist < 0.05 and 0.5 < area_ratio < 2.0:
                pairs.append((vi, ci))

        return pairs

    def _match_cached_boxes_to_current(
        self,
        cached_regions: np.ndarray,
        current_boxes: torch.Tensor,
        iou_threshold: float = 0.5,
        img_size: tuple = None,
        cached_img_size: tuple = None,
    ):
        if cached_regions is None or len(cached_regions) == 0 or len(current_boxes) == 0:
            return []

        cached_t = torch.as_tensor(cached_regions, dtype=torch.float32)
        current_t = current_boxes.float().cpu()

        if img_size is not None and cached_img_size is not None:
            w, h = img_size
            cw, ch = cached_img_size
            cached_t = cached_t / torch.tensor([cw, ch, cw, ch])
            current_t = current_t / torch.tensor([w, h, w, h])
        elif img_size is not None:
            w, h = img_size
            current_t = current_t / torch.tensor([w, h, w, h])

        def iou_1d(a, b):
            inter_x1 = torch.max(a[:, 0].unsqueeze(1), b[:, 0].unsqueeze(0))
            inter_y1 = torch.max(a[:, 1].unsqueeze(1), b[:, 1].unsqueeze(0))
            inter_x2 = torch.min(a[:, 2].unsqueeze(1), b[:, 2].unsqueeze(0))
            inter_y2 = torch.min(a[:, 3].unsqueeze(1), b[:, 3].unsqueeze(0))
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
            area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            union_area = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter_area
            return inter_area / union_area.clamp(min=1e-6)

        iou_matrix = iou_1d(cached_t, current_t)
        pairs = []
        used_current = set()
        for m in range(len(cached_t)):
            best_n = iou_matrix[m].argmax().item()
            if iou_matrix[m, best_n].item() >= iou_threshold and best_n not in used_current:
                pairs.append((best_n, m))
                used_current.add(best_n)
        return pairs
 
    def _get_cache_path(self, image_id: str) -> str:
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"{image_id}.pkl")
 
    def prepare_vlrm_regions(self, full_image: Image.Image, boxes: torch.Tensor, min_crop_size: int = 10) -> Tuple[List[Image.Image], List[int]]:
        """Strictly enforces region-based cropping with adaptive padding for small objects."""
        crops, valid_indices = [], []
        img_w, img_h = full_image.width, full_image.height
        for orig_idx, roi in enumerate(boxes):
            x1, y1, x2, y2 = map(float, roi.tolist())
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            # Mod 3: Small-object context expansion
            pad_scale = 1.0
            if area < 32 * 32:
                pad_scale = 1.40
            elif area < 96 * 96:
                pad_scale = 1.20
                
            cx, cy = x1 + w / 2, y1 + h / 2
            new_w, new_h = w * pad_scale, h * pad_scale
            
            x1 = max(0, int(cx - new_w / 2))
            y1 = max(0, int(cy - new_h / 2))
            x2 = min(img_w, int(cx + new_w / 2))
            y2 = min(img_h, int(cy + new_h / 2))
            
            if (x2-x1) < min_crop_size or (y2-y1) < min_crop_size: 
                continue
            crop = full_image.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop)
            valid_indices.append(orig_idx)
        return crops, valid_indices

    def _generate_vlrm_captions_modular(
        self, crops: List[Image.Image]
    ) -> List[str]:
        if len(crops) == 0:
            return []
        self.vlrm.prepare()
        
        # Batched caption generation to speed up inference significantly
        batch_size = 4
        captions = []
        for idx in range(0, len(crops), batch_size):
            batch_crops = crops[idx : idx + batch_size]
            batch_caps = self.vlrm.generate_captions_batch(batch_crops)
            captions.extend(batch_caps)
            
        self.vlrm.finish(force_offload=False)
        return captions

    def _apply_lpc_logic(self, semantic_scores, image_features, mean_features, image_id, tracker):
        """Internal helper to apply LPC with fail-safe behavior and logging."""
        try:
            if tracker: t_start_lpc = tracker.log_module_start("LPC MODULE", input_size=image_features.shape)
            print(f"[LPC] Applied: YES")
            
            lpc = LowDensityPriorConstraint(bandwidth=0.8)
            lpc_out = lpc.apply(semantic_scores, image_features, mean_features)
            
            lpc_weights = lpc_out["lpc_weights"]
            adjusted_scores = lpc_out["lpc_adjusted_scores"]
            
            # LPC Debug Visibility
            num_unknown = lpc_out["unknown_flags"].sum().item()
            density_vals = lpc_out["density_values"]
            before_mean = lpc_out["original_scores"].mean().item()
            after_mean = lpc_out["lpc_adjusted_scores"].mean().item()
            
            print(f"  [LPC MODULE]")
            print(f"    - Input proposals: {len(lpc_weights)}")
            print(f"    - Unknown detected: {num_unknown}")
            print(f"    - Density stats: min={density_vals.min():.3f} / max={density_vals.max():.3f} / mean={density_vals.mean():.3f}")
            print(f"    - Score shift summary: {before_mean:.4f} → {after_mean:.4f}")
            
            if tracker:
                tracker.save_debug_json("lpc", image_id, lpc_out)
                tracker.log_module_end("LPC MODULE", t_start_lpc, metadata={
                    "lpc_stats": {
                        "low_density_count": num_unknown,
                        "weight_mean": lpc_weights.mean().item()
                    },
                    "score_delta": {
                        "before_mean": before_mean,
                        "after_mean": after_mean,
                        "change_percent": (after_mean - before_mean) / (before_mean + 1e-6) * 100
                    }
                })
            return adjusted_scores
        except Exception as e:
            print(f"[LPC ERROR] Module failed: {str(e)}")
            print("[LPC] Continuing pipeline using Mahalanobis output (Fail-safe).")
            return semantic_scores
 
    def run_stage_2(
        self,
        background_rois: torch.Tensor,
        full_image: Image.Image,
        class_names: List[str],
        image_id: str = None,
        image_path: str = None,
        max_regions: int = 50,
        min_crop_size: int = 10,
        force_vlrm_rerun: bool = False,
        verbose: bool = False,
        tracker: Any = None,
        return_raw_features: bool = False,
    ) -> Any:
        num_classes = len(class_names)
        
        # 3. VALIDATION CHECK BEFORE VLRM
        assert len(background_rois) > 0, "run_stage_2 called with 0 background proposals. This should be skipped earlier."
        
        num_rois = len(background_rois)
        areas = (background_rois[:, 2] - background_rois[:, 0]).float() * \
                (background_rois[:, 3] - background_rois[:, 1]).float()

        # Balanced ROI selection: mix large objects (by area) + small objects (rare LVIS).
        # Area-only sorting suppresses small rare objects. This fixes Bug #2.
        if num_rois <= max_regions:
            sorted_indices = torch.arange(num_rois, device=background_rois.device)
        else:
            half = max_regions // 2
            # Top half by area (large objects — easier to classify)
            top_large = torch.argsort(areas, descending=True)[:half]
            # Bottom half by area (small/rare objects — LVIS-critical)
            top_small = torch.argsort(areas, descending=False)[:half]
            combined = torch.cat([top_large, top_small]).unique()
            sorted_indices = combined[:max_regions]

        # Re-index background_rois to process the selected boxes
        filtered_rois = background_rois[sorted_indices]
        
        # 9. CLEAN ARCHITECTURE - Encapsulate input preparation
        crops, local_valid_indices = self.prepare_vlrm_regions(full_image, filtered_rois, min_crop_size)
        
        # Map local indices back to original indices
        valid_indices = [sorted_indices[i].item() for i in local_valid_indices]
 
        config = {"cache_version": "v2", "max_regions": max_regions, "min_crop_size": min_crop_size, "vlrm_model": self.vlrm.model_name}
        input_hash = self._compute_input_hash(image_path, config) if image_path else ""
        cache_path = self._get_cache_path(image_id) if image_id else None
        cache_data, mode = None, "EXECUTED"

        if tracker: t_start = tracker.log_module_start("Stage-2-VLRM", input_size=background_rois.shape)

        # ── Cache loading ──────────────────────────────────────────────────────
        # We do NOT validate by hash because existing cache files were saved
        # with an old hash format (included background_rois bytes) that can never
        # match the new format (image_path + config only).
        # Instead we use image_id as the unique cache key and reconstruct
        # valid_indices by IoU-matching cached box coords to current boxes.
        # ──────────────────────────────────────────────────────────────────────
        if cache_path and os.path.exists(cache_path) and not force_vlrm_rerun:
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                cached_captions = cache_data.get("captions", [])
                cached_regions  = cache_data.get("regions", None)
                cached_valid_indices = cache_data.get("valid_indices", [])
                cached_image_size = cache_data.get("image_size", None)

                if not isinstance(cached_captions, list):
                    cache_data = None
                else:
                    mode = "LOADED FROM CACHE"

                    img_size = (full_image.width, full_image.height)

                    pairs = None
                    if cached_regions is not None and len(cached_valid_indices) > 0:
                        pairs = self._match_cached_indices(
                            cached_regions, cached_valid_indices,
                            background_rois, img_size,
                        )

                    if pairs is not None and len(pairs) == 0 and cached_regions is not None:
                        pairs = self._match_cached_boxes_to_current(
                            cached_regions, background_rois, iou_threshold=0.5,
                            img_size=img_size, cached_img_size=cached_image_size,
                        )

                    if pairs is not None and len(pairs) > 0:
                        captions = [cached_captions[ci] for (_, ci) in pairs]
                        valid_indices = [cur_idx for (cur_idx, _) in pairs]
                        crops = [
                            full_image.crop((
                                max(0, int(background_rois[vi, 0])),
                                max(0, int(background_rois[vi, 1])),
                                min(full_image.width,  int(background_rois[vi, 2])),
                                min(full_image.height, int(background_rois[vi, 3]))
                            )).convert("RGB")
                            for vi in valid_indices
                        ]
                        print(f"  [VLRM] ✓ Cache hit — {len(captions)}/{len(cached_captions)} regions reused for image_id={image_id}")
                    else:
                        captions = cached_captions
                        valid_indices = cached_valid_indices
                        max_idx = len(background_rois) - 1
                        valid_indices = [vi for vi in valid_indices if vi <= max_idx]
                        captions = captions[:len(valid_indices)]
                        crops = [
                            full_image.crop((
                                max(0, int(background_rois[vi, 0])),
                                max(0, int(background_rois[vi, 1])),
                                min(full_image.width,  int(background_rois[vi, 2])),
                                min(full_image.height, int(background_rois[vi, 3]))
                            )).convert("RGB")
                            for vi in valid_indices
                        ]
                        print(f"  [VLRM] ✓ Cache hit — {len(captions)}/{len(cached_captions)} regions reused for image_id={image_id}")

            except Exception as e:
                print(f"[VLRM ERROR] Failed to load cache: {str(e)}. Deleting corrupted file.")
                cache_data = None
                if cache_path and os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

        if cache_data is None:
            print(f"  [VLRM] Cache miss. Generating captions for {len(crops)} proposals...")
            captions = self._generate_vlrm_captions_modular(crops)
            if cache_path:
                temp_path = cache_path + ".tmp"
                try:
                    with open(temp_path, "wb") as f:
                        pickle.dump({
                            "image_id": image_id,
                            "regions": background_rois[valid_indices].cpu().numpy(),
                            "captions": captions,
                            "input_hash": input_hash,
                            "num_regions": len(captions),
                            "valid_indices": valid_indices,
                            "image_size": (full_image.width, full_image.height),
                            "timestamp": datetime.now().isoformat()
                        }, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temp_path, cache_path)
                    print(f"  [VLRM] ✓ Saved {len(captions)} captions to {os.path.basename(cache_path)}")
                except Exception as e:
                    print(f"[VLRM ERROR] Failed to save cache: {str(e)}")
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass
 
        if tracker:
            cap_lens = [len(c) for c in captions] if captions else [0]
            tracker.log_module_end("Stage-2-VLRM", t_start, status=mode, metadata={"num_regions": len(captions), "sample_captions": captions[:3], "caption_stats": {"mean_len": sum(cap_lens)/max(1, len(cap_lens))}})
 
        # Quality Filter: filter out generic/background captions while retaining real object descriptions
        valid_caption_mask = torch.ones(len(captions), dtype=torch.bool, device=self.device)
        generic_words = {"background", "scene", "wall", "floor", "ceiling", "texture", "pattern", "blurry", "blur", "view", "area", "surface", "part", "piece", "section"}
        
        for ci, cap in enumerate(captions):
            cap_lower = cap.lower().strip()
            
            # Remove common templates to check the core content of the caption
            prefixes = [
                "a photo of a ", "a photo of ", "photo of a ", "photo of ",
                "an image of a ", "an image of ", "image of a ", "image of ",
                "a picture of a ", "a picture of ", "picture of a ", "picture of ",
                "closeup of a ", "closeup of ", "close up of a ", "close up of "
            ]
            core_content = cap_lower
            for prefix in prefixes:
                if core_content.startswith(prefix):
                    core_content = core_content[len(prefix):].strip()
                    break
            
            # If the core content is empty or too short (e.g. just prefix, or single letters)
            if not core_content or len(core_content) < 3:
                valid_caption_mask[ci] = False
                continue
                
            # If the core content is a background/generic word, or if the caption contains background words
            is_generic = core_content in generic_words or any(word in core_content for word in {"background", "blurry"})
            # Reject if the core content is just placeholder words like "image", "photo", "picture", "object"
            if is_generic or core_content in {"image", "photo", "picture", "object", "something", "an object"}:
                valid_caption_mask[ci] = False
                
        # Build filtered index mapping for downstream code
        valid_caption_indices = torch.where(valid_caption_mask)[0].tolist()  # local indices

        if not captions:
            if return_raw_features:
                dim = 512
                if hasattr(self.clip.model, 'parameters'):
                    for p in self.clip.model.parameters():
                        if len(p.shape) > 0:
                            dim = p.shape[-1]
                            break
                return torch.empty(0, dim, device=self.device), torch.empty(0, dim, device=self.device), []
            return torch.empty(0, num_classes, device=self.device), []
 
        if tracker: t_start_clip = tracker.log_module_start("Stage-2-CLIP")
        caption_features = self.clip.encode_captions_batch(captions)
        image_features = self.clip.encode_images_batch(crops)
        if tracker: tracker.log_module_end("Stage-2-CLIP", t_start_clip, metadata={"embedding_shape": caption_features.shape})

        # Filter out invalid captions instead of zero-ing them.
        # Zero vectors have undefined/extreme Mahalanobis distance (Bug #9/#10).
        if valid_caption_mask.any() and not valid_caption_mask.all():
            caption_features = caption_features[valid_caption_mask]
            image_features = image_features[valid_caption_mask]
            valid_indices = [valid_indices[i] for i in valid_caption_indices]
            print(f"  [VLRM] Filtered {(~valid_caption_mask).sum().item()} invalid captions. {len(valid_indices)} remain.")

        if return_raw_features:
            return caption_features, image_features, valid_indices

        mean_features, cov_inv = self._get_class_stats(class_names)
 
        if tracker: t_start_mahalanobis = tracker.log_module_start("Stage-2-Mahalanobis")
        caption_scores_raw = self.clip.compute_mahalanobis(caption_features, mean_features, cov_inv)
        image_scores_raw = self.clip.compute_mahalanobis(image_features, mean_features, cov_inv)

        # Per-class temperature scaling removed (Bug #4): it distorts the Mahalanobis
        # metric space and inverts the calibration intent for rare classes.
        # Apply softmax directly on the Mahalanobis scores.
        caption_sim = torch.softmax(caption_scores_raw, dim=-1)
        image_sim = torch.softmax(image_scores_raw, dim=-1)

        semantic_scores = self.fusion_alpha * caption_sim + (1 - self.fusion_alpha) * image_sim
        if tracker: tracker.log_module_end("Stage-2-Mahalanobis", t_start_mahalanobis)
        
        if self.use_lpc:
            semantic_scores = self._apply_lpc_logic(semantic_scores, image_features, mean_features, image_id, tracker)
        else:
            print(f"[LPC] Applied: NO (Disabled for this pipeline)")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        return semantic_scores, valid_indices
 
if __name__ == "__main__":
    # Test script...
    pass
