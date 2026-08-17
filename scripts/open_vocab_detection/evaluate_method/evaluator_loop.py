"""
FYP: Evaluator Loop - 3-Stage Object Detection Pipeline
=========================================================

Stage 1: GDINO + Mask-RCNN in parallel (UNCHANGED)
Stage 2: VLRM + CLIP semantic reasoning (NEW)
Stage 3: Concatenation + SRM + SAM + NMS (CORRECTED)
"""
import torch
import numpy as np
import time
import json
import os
import sys
import logging
import cv2
import gc

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from PIL import Image

# Add parent scripts directory for modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from vlrm_clip_module import Stage2_VLRM_CLIP_Fusion
from srm_module import ScoreRefinementModule

from ground_dino_utils import (
    inference_gdino, 
    inference_maskrcnn, 
    extract_background_proposals
)
from detectron2.structures import Instances, Boxes


def _flush():
    """Force stdout flush for real-time output visibility."""
    sys.stdout.flush()
    sys.stderr.flush()

@torch.no_grad()
def inference(data_loader, evaluator, model, text_prompt, param_dict):
    evaluator.reset()
    _run_generic_evaluation_loop(data_loader, evaluator, model, text_prompt, param_dict)
    
    results = evaluator.evaluate()
    if results is None:
        results = {}

    return results


def _run_generic_evaluation_loop(data_loader, evaluator, model, text_prompt, param_dict):
    logger = logging.getLogger(__name__)
    logger.info("###\n### Start inference on {} batches\n###".format(len(data_loader)))
    
    # --- Model and parameter extraction ---
    maskrcnn_model = param_dict.get("maskrcnn_model")
    sam = param_dict.get("sam")
    device = param_dict.get("device", "cuda")
    coco_classes = param_dict.get("coco_classes", [])
    verbose = param_dict.get("verbose", False)
    
    # --- Initialize Stage-2 Processor (VLRM + CLIP) ---
    stage2_processor = Stage2_VLRM_CLIP_Fusion(device=device)
    stage2_enabled = maskrcnn_model is not None
    
    # --- Initialize Stage-3 SRM ---
    srm = ScoreRefinementModule(per_image_norm=True)
    
    if stage2_enabled:
        print("[Pipeline] Stage-2 ENABLED: VLRM + CLIP Fusion")
        print("[Pipeline] Stage-3: Concatenation + SRM + SAM + NMS")
    else:
        print("[Pipeline] Stage-2 DISABLED: GDINO-only mode")
    _flush()

    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()

    for idx, inputs in enumerate(tqdm(data_loader)):
        if idx == num_warmup:
            start_time = time.perf_counter()

        image_start = time.perf_counter()
        image_name = os.path.basename(inputs[0].get('file_name', f'image_{idx}'))
        print(f"\n{'='*70}")
        print(f"[Image {idx+1}/{total}] {image_name}")
        print(f"{'='*70}")
        _flush()
        
        # =====================================================================
        # STAGE 1: GDINO + Mask-RCNN (UNCHANGED)
        # =====================================================================
        stage1_start = time.perf_counter()
        gdino_outputs = inference_gdino(model, inputs, text_prompt, param_dict)
        gdino_instances = gdino_outputs[0]["instances"]
        stage1_time = time.perf_counter() - stage1_start
        print(f"[Stage 1] GDINO: {len(gdino_instances)} detections ({stage1_time:.2f}s)")
        
        # Initialize final instances with Stage-1 results.
        final_instances = gdino_instances

        if stage2_enabled:
            try:
                m_device = next(maskrcnn_model.parameters()).device
            except:
                m_device = device
                
            maskrcnn_boxes, maskrcnn_scores, maskrcnn_classes, known_boxes, known_scores, known_classes = inference_maskrcnn(
                maskrcnn_model, inputs, m_device
            )
            print(f"[Stage 1] Mask-RCNN: {len(maskrcnn_boxes)} RPN proposals")
            
            # Map known classes from TorchVision (1-91) to COCO contiguous (0-79)
            if len(known_classes) > 0:
                tv_to_contig = {
                    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
                    13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18, 21: 19,
                    22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27, 33: 28,
                    34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37,
                    43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46,
                    53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55,
                    62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63, 74: 64,
                    75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72, 84: 73,
                    85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79
                }
                valid_mask = torch.zeros_like(known_classes, dtype=torch.bool)
                for i in range(len(known_classes)):
                    lbl = known_classes[i].item()
                    if lbl in tv_to_contig:
                        known_classes[i] = tv_to_contig[lbl]
                        valid_mask[i] = True
                known_boxes = known_boxes[valid_mask]
                known_scores = known_scores[valid_mask]
                known_classes = known_classes[valid_mask]
            
            _flush()
            
            # =================================================================
            # STAGE 2: Background Proposal Labeling (VLRM + CLIP)
            # =================================================================
            labeled_boxes = torch.empty(0, 4)
            labeled_scores = torch.empty(0)
            labeled_classes = torch.empty(0, dtype=torch.int64)
            
            if len(maskrcnn_boxes) > 0:
                bg_boxes, _, _ = extract_background_proposals(
                    gdino_instances.pred_boxes if len(gdino_instances) > 0 else None,
                    maskrcnn_boxes, maskrcnn_scores, maskrcnn_classes, iou_threshold=0.5
                )
                print(f"[Stage 2] After IoU filtering: {len(bg_boxes)} background proposals")
                
                if len(bg_boxes) > 0:
                    stage2_start = time.perf_counter()
                    pil_image = Image.open(inputs[0]["file_name"]).convert("RGB")

                    # Offload GDINO and SAM to CPU to free GPU for VLRM
                    model.to("cpu")
                    if sam is not None:
                        sam.to("cpu")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    semantic_scores, valid_indices = stage2_processor.run_stage_2(
                        background_rois=bg_boxes, 
                        full_image=pil_image, 
                        class_names=coco_classes,
                        verbose=verbose,
                    )

                    if semantic_scores.numel() > 0 and len(valid_indices) > 0:
                        confidence_scores, class_indices = torch.max(semantic_scores, dim=1)
                        # Raw cosine similarities: good matches are typically 0.20-0.40
                        CONF_THRESHOLD = 0.20
                        keep_mask = confidence_scores > CONF_THRESHOLD
                        keep_local = torch.nonzero(keep_mask).squeeze(1)
                        # Safeguard against list index out of range if shapes mismatch
                        keep_local = keep_local[keep_local < len(valid_indices)]
                        
                        if keep_local.numel() > 0:
                            # Map back to original bg_boxes indices via valid_indices
                            original_indices = [valid_indices[k] for k in keep_local.tolist()]
                            labeled_boxes = bg_boxes[original_indices]
                            labeled_scores = confidence_scores[keep_local]
                            labeled_classes = class_indices[keep_local]
                    
                    stage2_time = time.perf_counter() - stage2_start
                    print(f"[Stage 2] VLRM+CLIP processed {len(bg_boxes)} proposals → {len(labeled_boxes)} labeled ({stage2_time:.2f}s)")
                    
                    # Restore GDINO and SAM to GPU
                    model.to(device)
                    if sam is not None:
                        sam.to(device)
                    torch.cuda.empty_cache()
            
            print(f"[Stage 2] Labeled {len(labeled_boxes)} background objects")
            _flush()
            
            # Combine Stage-2 labeled background objects with Stage-1 known objects
            if len(known_boxes) > 0:
                print(f"[Stage 2] Incorporating {len(known_boxes)} Mask-RCNN known objects into fusion")
            
            combined_boxes = torch.cat([labeled_boxes, known_boxes.to(labeled_boxes.device)], dim=0) if len(labeled_boxes) > 0 or len(known_boxes) > 0 else torch.empty(0, 4, device=labeled_boxes.device)
            combined_scores = torch.cat([labeled_scores, known_scores.to(labeled_scores.device)], dim=0) if len(labeled_scores) > 0 or len(known_scores) > 0 else torch.empty(0, device=labeled_scores.device)
            combined_classes = torch.cat([labeled_classes, known_classes.to(labeled_classes.device)], dim=0) if len(labeled_classes) > 0 or len(known_classes) > 0 else torch.empty(0, dtype=torch.int64, device=labeled_classes.device)
            
            # =================================================================
            # STAGE 3: Concatenation + SRM + SAM + NMS (UNCHANGED LOGIC)
            # =================================================================
            if len(combined_boxes) > 0:
                final_instances = _merge_detections_srm(
                    gdino_instances, combined_boxes, combined_scores, combined_classes,
                    inputs[0]["height"], inputs[0]["width"],
                    sam=sam, srm=srm, param_dict=param_dict,
                    image_file=inputs[0]["file_name"],
                )
        
        outputs = [{"instances": final_instances}]
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        evaluator.process(inputs, outputs)
        
        image_time = time.perf_counter() - image_start
        n_final = len(outputs[0]["instances"])
        print(f"[Summary] Image {idx+1}/{total}: {n_final} final detections | {image_time:.2f}s")
        _flush()

        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

    total_time = time.perf_counter() - start_time
    import datetime
    total_time_str = str(datetime.timedelta(seconds=total_time))
    avg_time = total_time / max(1, total - num_warmup)
    logger.info("Total inference time: {} ({:.6f} s / iter per device)".format(
        total_time_str, avg_time
    ))
    print(f"\n{'='*70}")
    print(f"Pipeline Complete!")
    print(f"  Total time:    {total_time_str}")
    print(f"  Avg per image: {avg_time:.2f}s")
    print(f"  Images:        {total}")
    print(f"{'='*70}")
    _flush()


# ==============================================================================
# Stage-3: Concatenation + SRM + SAM + NMS
# ==============================================================================

def _merge_detections_srm(
    gdino_instances, 
    stage2_boxes, stage2_scores, stage2_classes, 
    height, width,
    sam=None, srm=None, param_dict=None,
    image_file=None,
):
    
    from torchvision.ops import batched_nms
    from segment_anything.utils.transforms import ResizeLongestSide
    
    verbose = param_dict.get('verbose', False) if param_dict else False
    
    if len(stage2_boxes) == 0:
        return gdino_instances
    
    # --- Step 1: OT-Based Fusion (replaces simple concatenation) ---
    from ot_fusion_module import ot_fusion
    
    if len(gdino_instances) > 0:
        gdino_boxes = gdino_instances.pred_boxes.tensor.cpu()
        gdino_scores = gdino_instances.scores.cpu()
        gdino_classes = gdino_instances.pred_classes.cpu()
    else:
        gdino_boxes = torch.empty(0, 4)
        gdino_scores = torch.empty(0)
        gdino_classes = torch.empty(0, dtype=torch.int64)
    
    stage2_boxes = stage2_boxes.cpu() if torch.is_tensor(stage2_boxes) else torch.tensor(stage2_boxes)
    stage2_scores = stage2_scores.cpu() if torch.is_tensor(stage2_scores) else torch.tensor(stage2_scores, dtype=torch.float32)
    stage2_classes = stage2_classes.cpu() if torch.is_tensor(stage2_classes) else torch.tensor(stage2_classes, dtype=torch.int64)
    
    # Clamp Stage-2 classes to valid range [0, 79]
    stage2_classes = torch.clamp(stage2_classes, min=0, max=79)
    
    # OT fusion: re-weights scores based on cross-source IoU and semantic consistency
    # via Sinkhorn optimal transport. Boxes and classes pass through unchanged.
    merged_boxes, merged_scores, merged_classes = ot_fusion(
        gdino_boxes, gdino_scores, gdino_classes,
        stage2_boxes, stage2_scores, stage2_classes,
        iou_weight=0.6,
        semantic_weight=0.4,
        sinkhorn_reg=0.1,
        sinkhorn_iters=50,
        param_dict={'verbose': verbose}
    )
    
    if verbose:
        print(f"[Stage 3] OT Fusion: GDINO={len(gdino_boxes)}, Stage-2={len(stage2_boxes)}, Total={len(merged_boxes)}")
    
    # --- Step 2: SAM mask quality scores ---
    sam_scores = None
    if sam is not None and image_file is not None and len(merged_boxes) > 0:
        try:
            resize_transform = param_dict.get("resize_transform")
            if resize_transform is None:
                resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
            
            curr_image = cv2.imread(image_file)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            img_shape = curr_image.shape[:2]
            curr_image = resize_transform.apply_image(curr_image)
            curr_image = torch.as_tensor(curr_image, device=sam.device).permute(2, 0, 1).contiguous()
            
            boxes_on_device = merged_boxes.to(sam.device)
            sam_box_prompts = resize_transform.apply_boxes_torch(boxes_on_device, img_shape)
            
            # Process SAM in batches
            sam_batch_size = 50
            sam_scores_list = []
            sam_refined_boxes_list = []
            
            for i in range(0, len(sam_box_prompts), sam_batch_size):
                batch_prompts = sam_box_prompts[i:i + sam_batch_size]
                if len(batch_prompts) == 0:
                    continue
                
                from ground_dino_utils import batched_mask_to_box
                
                batched_input = [{
                    "image": curr_image,
                    "boxes": batch_prompts,
                    "original_size": img_shape,
                }]
                
                batched_output = sam(batched_input, multimask_output=False)
                
                sam_refined_boxes_list.append(
                    batched_mask_to_box(batched_output[0]['masks'].clone().detach()).squeeze(1)
                )
                sam_scores_list.append(
                    batched_output[0]['iou_predictions']
                )
                
                del batched_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if len(sam_scores_list) > 0:
                sam_scores = torch.cat(sam_scores_list, dim=0).squeeze(1).cpu()
                sam_refined_boxes = torch.cat(sam_refined_boxes_list, dim=0).cpu()
                # Use SAM-refined boxes
                merged_boxes = sam_refined_boxes
                
                if verbose:
                    print(f"[Stage 3] SAM: {len(sam_scores)} mask quality scores, range=[{sam_scores.min():.3f}, {sam_scores.max():.3f}]")
        
        except Exception as e:
            print(f"[Stage 3] SAM error (continuing without): {e}")
            sam_scores = None
    
    # --- Step 3: SRM refinement ---
    if srm is not None and sam_scores is not None and len(sam_scores) > 0:
        print(f"[SRM] Refining {len(merged_scores)} scores using SAM mask quality...")
        merged_scores = srm.refine_scores(merged_scores, sam_scores)
        print(f"[SRM] ✓ Scores refined: range=[{merged_scores.min():.3f}, {merged_scores.max():.3f}]")
    
    # --- Step 4: NMS ---
    if len(merged_boxes) > 0:
        # Ensure correct dtypes — NMS requires float boxes/scores, long classes
        merged_boxes = merged_boxes.float()
        merged_scores = merged_scores.float()
        merged_classes = merged_classes.long()
        
        keep = batched_nms(merged_boxes, merged_scores, merged_classes, iou_threshold=0.5)
        merged_boxes = merged_boxes[keep]
        merged_scores = merged_scores[keep]
        merged_classes = merged_classes[keep]
    
    # Keep top 100
    if len(merged_scores) > 100:
        topk_scores, topk_idxs = torch.topk(merged_scores, 100)
        merged_boxes = merged_boxes[topk_idxs]
        merged_scores = topk_scores
        merged_classes = merged_classes[topk_idxs]
    
    # Build merged Instances
    merged = Instances((height, width))
    merged.pred_boxes = Boxes(merged_boxes)
    merged.scores = merged_scores
    merged.pred_classes = merged_classes
    
    print(f"[Stage 3] Final: GDINO={len(gdino_instances)}, Stage-2={len(stage2_boxes)}, After SRM+NMS={len(merged)}")
    _flush()
    
    return merged
