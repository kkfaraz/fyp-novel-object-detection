import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time

def download_sam3_checkpoint(checkpoint_dir="SAM_weights"):
    import os
    import shutil
    from huggingface_hub import hf_hub_download
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    target_path = os.path.join(checkpoint_dir, "sam3.pt")
    
    # Check if a valid model already exists
    if os.path.exists(target_path):
        size_gb = os.path.getsize(target_path) / (1024**3)
        if size_gb > 3.0:
            print(f"[SAM3 Loader] Found existing checkpoint at {target_path} ({size_gb:.2f} GB)")
            return target_path
        else:
            print(f"[SAM3 Loader] Existing checkpoint at {target_path} is too small ({size_gb:.4f} GB). Re-downloading...")
            try:
                os.remove(target_path)
            except OSError:
                pass

    print("[SAM3 Loader] sam3.pt not found. Starting download...")
    
    # Attempt 1: Official gated repository
    try:
        print("[SAM3 Loader] Attempting to download from official facebook/sam3...")
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            cache_dir=checkpoint_dir,
            resume_download=True
        )
        shutil.copy(checkpoint_path, target_path)
        print(f"[SAM3 Loader] Official download complete. Saved to {target_path}")
        return target_path
    except Exception as e:
        print(f"[SAM3 Loader] Official download failed: {e}. Trying public mirror...")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
                
    # Attempt 2: Un-gated mirror repository
    try:
        print("[SAM3 Loader] Attempting to download from mirror 1038lab/sam3...")
        checkpoint_path = hf_hub_download(
            repo_id="1038lab/sam3",
            filename="sam3.pt",
            cache_dir=checkpoint_dir,
            resume_download=True
        )
        shutil.copy(checkpoint_path, target_path)
        print(f"[SAM3 Loader] Mirror download complete. Saved to {target_path}")
        return target_path
    except Exception as e2:
        print(f"[SAM3 Loader] Mirror download failed: {e2}")
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        raise e2

class SAM3CallableWrapper(nn.Module):
    def __init__(self, sam3_model, device):
        super().__init__()
        self.model = sam3_model
        self.device = device
        self.predictor = self.model.inst_interactive_predictor
        if self.predictor is None:
            raise ValueError("inst_interactive_predictor is None. Make sure enable_inst_interactivity=True was passed to build_sam3_image_model.")
        
        # Caching image features to avoid redundant backbone execution on the same image
        self.current_image_tensor = None
        self.current_orig_hw = None
        
        # Expose image_encoder.img_size for compatibility with pipeline stages (e.g. ResizeLongestSide)
        self.image_encoder = type('DummyEncoder', (object,), {'img_size': self.predictor.model.image_size})

    @torch.no_grad()
    def forward(self, batched_input, multimask_output=False):
        # Time the inference call for telemetry/diagnostics
        t_start = time.time()
        
        assert len(batched_input) == 1, "Only single-image batching is supported by Stage-3 pipeline"
        item = batched_input[0]
        
        image_tensor = item["image"] # Shape (3, H_resized, W_resized) on device, range [0, 255]
        boxes_tensor = item["boxes"] # Shape (N, 4) on device, in 1008-resized space
        orig_h, orig_w = item["original_size"]
        
        # Verify device
        if image_tensor.device != self.device:
            image_tensor = image_tensor.to(self.device)
        if boxes_tensor.device != self.device:
            boxes_tensor = boxes_tensor.to(self.device)
            
        # Check if the image matches the cached one
        is_same_image = False
        if self.current_image_tensor is not None and self.current_image_tensor.shape == image_tensor.shape:
            # Quick check to avoid costly tensor equality checks
            if torch.equal(self.current_image_tensor[0, 0, :10], image_tensor[0, 0, :10]):
                is_same_image = True
                
        # Get GPU memory before forward pass
        mem_before = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
        
        if not is_same_image:
            self.current_image_tensor = image_tensor
            self.current_orig_hw = (orig_h, orig_w)
            
            # Normalization and interpolation on GPU
            img = image_tensor.float() / 255.0
            # Interpolate to the square 1008x1008 resolution
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=(self.predictor.model.image_size, self.predictor.model.image_size),
                mode="bilinear",
                align_corners=False
            )
            # Normalize to mean=0.5, std=0.5
            img = (img - 0.5) / 0.5
            
            # Forward image through SAM3 backbone (using self.model instead of self.predictor.model)
            backbone_out = self.model.backbone.forward_image(img)
            
            # Extract sam2 backbone output
            sam2_backbone_out = backbone_out["sam2_backbone_out"]
            
            # Apply conv_s0 and conv_s1 projections to match the expected format
            if hasattr(self.predictor.model.sam_mask_decoder, "conv_s0"):
                sam2_backbone_out["backbone_fpn"][0] = (
                    self.predictor.model.sam_mask_decoder.conv_s0(
                        sam2_backbone_out["backbone_fpn"][0]
                    )
                )
            if hasattr(self.predictor.model.sam_mask_decoder, "conv_s1"):
                sam2_backbone_out["backbone_fpn"][1] = (
                    self.predictor.model.sam_mask_decoder.conv_s1(
                        sam2_backbone_out["backbone_fpn"][1]
                    )
                )
            
            _, vision_feats, _, _ = self.predictor.model._prepare_backbone_features(sam2_backbone_out)
            vision_feats[-1] = vision_feats[-1] + self.predictor.model.no_mem_embed

            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], self.predictor._bb_feat_sizes[::-1])
            ][::-1]
            self.predictor._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            self.predictor._is_image_set = True
            self.predictor._orig_hw = [(orig_h, orig_w)]
            
        # Re-scale box coordinates from 1008-resized space to original image space
        scale = self.predictor.model.image_size / max(orig_h, orig_w)
        original_boxes = boxes_tensor / scale
        
        # Prepare prompts
        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
            point_coords=None,
            point_labels=None,
            box=original_boxes,
            mask_logits=None,
            normalize_coords=True,
            img_idx=0
        )
        
        # Run prediction
        masks, iou_predictions, low_res_masks = self.predictor._predict(
            point_coords=unnorm_coords,
            point_labels=labels,
            boxes=unnorm_box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=False,
            img_idx=0
        )
        
        mem_after = torch.cuda.memory_allocated(self.device) if torch.cuda.is_available() else 0
        t_end = time.time()
        
        # Logging Telemetry
        print(f"[SAM3 Telemetry] Box count: {len(boxes_tensor)}, Time: {t_end - t_start:.4f}s, "
              f"GPU Memory Delta: {(mem_after - mem_before)/1e6:.2f} MB")
              
        # Empty CUDA cache to maintain footprint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return [{
            "masks": masks.clone().detach(),
            "iou_predictions": iou_predictions.clone().detach()
        }]
