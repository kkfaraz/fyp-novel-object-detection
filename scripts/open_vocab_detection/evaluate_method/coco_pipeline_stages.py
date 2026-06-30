import torch
import time
import os
import sys
import gc
import cv2
import hashlib
from tqdm import tqdm
from PIL import Image

# Add parent scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from vlrm_clip_module import Stage2_VLRM_CLIP_Fusion
from srm_module import ScoreRefinementModule
from ground_dino_utils import (
    inference_gdino,
    inference_maskrcnn,
    extract_background_proposals,
    batched_mask_to_box
)
from ot_fusion_module import ot_fusion
from detectron2.structures import Instances, Boxes
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import batched_nms
from pipeline_utils import PipelineTracker
from gpu_utils import log_gpu_memory

def _fix_fname(fname):
    """
    Fix absolute path mismatch when loading checkpoints from a different machine or directory.
    Replaces the prefix before 'DETECTRON2_DATASETS' with the current project path.
    """
    if "DETECTRON2_DATASETS" in fname:
        proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        parts = fname.split("DETECTRON2_DATASETS")
        return os.path.join(proj_path, "datasets", "DETECTRON2_DATASETS" + parts[-1])
    return fname

def run_stage1_coco(dataloader, model, text_prompt, param_dict):
    """
    Run Stage 1: GDINO + Mask-RCNN (if enabled).
    Returns list of dicts.
    """
    print("[COCO Stage 1] Running GDINO + Mask-RCNN on dataloader...")
    
    results = []
    
    maskrcnn_model = param_dict.get("maskrcnn_model")
    stage2_enabled = maskrcnn_model is not None
    device = param_dict.get("device", "cuda")
    
    debug_mode = param_dict.get("debug_mode", True)
    verbose_level = param_dict.get("verbose_level", "medium")
    out_dir = param_dict.get("out_dir", "debug_outputs")
    tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, "debug_outputs"))
    param_dict["tracker"] = tracker
    
    for idx, inputs in enumerate(tqdm(dataloader, desc="Stage 1")):
        image_file = inputs[0]["file_name"]
        image_id = inputs[0].get("image_id", idx) 
        
        # GDINO
        t_start_gdino = tracker.log_module_start("GDINO")
        gdino_outputs = inference_gdino(model, inputs, text_prompt, param_dict)
        gdino_instances = gdino_outputs[0]["instances"]
        num_gdino = len(gdino_instances)
        tracker.log_module_end("GDINO", t_start_gdino, metadata={
            "num_boxes": num_gdino,
            "score_stats": {"min": gdino_instances.scores.min().item() if num_gdino > 0 else 0,
                            "max": gdino_instances.scores.max().item() if num_gdino > 0 else 0,
                            "mean": gdino_instances.scores.mean().item() if num_gdino > 0 else 0}
        })
        
        if debug_mode and num_gdino > 0:
            tracker.save_debug_image("GDINO", image_file, boxes=gdino_instances.pred_boxes.tensor[:10])
        
        maskrcnn_out = {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "classes": torch.empty(0, dtype=torch.int64, device=device)
        }
        
        if stage2_enabled:
            t_start_rcnn = tracker.log_module_start("Mask-RCNN")
            try:
                m_device = next(maskrcnn_model.parameters()).device
            except:
                m_device = device
                
            m_boxes, m_scores, m_classes, known_boxes, known_scores, known_classes = inference_maskrcnn(
                maskrcnn_model, inputs, m_device
            )
            maskrcnn_out["boxes"] = m_boxes.to(device)
            maskrcnn_out["scores"] = m_scores.to(device)
            maskrcnn_out["classes"] = m_classes.to(device)
            maskrcnn_out["known_boxes"] = known_boxes.to(device)
            maskrcnn_out["known_scores"] = known_scores.to(device)
            maskrcnn_out["known_classes"] = known_classes.to(device)
            
            num_rcnn = len(m_boxes)
            tracker.log_module_end("Mask-RCNN", t_start_rcnn, metadata={
                "num_boxes": num_rcnn,
                "score_stats": {"min": m_scores.min().item() if num_rcnn > 0 else 0,
                                "max": m_scores.max().item() if num_rcnn > 0 else 0,
                                "mean": m_scores.mean().item() if num_rcnn > 0 else 0}
            })
            
        res_item = {
            "index": idx,
            "image_id": image_id,
            "file_name": image_file,
            "height": inputs[0]["height"],
            "width": inputs[0]["width"],
            "gdino_boxes": gdino_instances.pred_boxes.tensor.to(device),
            "gdino_scores": gdino_instances.scores.to(device),
            "gdino_classes": gdino_instances.pred_classes.to(device),
            "maskrcnn": maskrcnn_out
        }
        results.append(res_item)
        
    log_gpu_memory("after_stage1_coco")
    return results

def run_stage2_coco(stage1_results, param_dict):
    """
    Run Stage 2: VLRM + CLIP on background proposals.
    """
    print("[COCO Stage 2] Running VLRM + CLIP...")
    
    # Clear GPU memory from Stage 1 before initializing Stage 2 models
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tracker = param_dict.get("tracker")
    if tracker is None:
        debug_mode = param_dict.get("debug_mode", True)
        verbose_level = param_dict.get("verbose_level", "medium")
        out_dir = param_dict.get("out_dir", "debug_outputs")
        tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, "debug_outputs"))
        param_dict["tracker"] = tracker
        
    device = param_dict.get("device", "cuda")
    coco_classes = param_dict.get("coco_classes", [])
    verbose = param_dict.get("verbose", False)
    force_vlrm_rerun = param_dict.get("force_vlrm_rerun", False)
    
    # Initialize VLRM
    vlrm_cache_dir = param_dict.get("vlrm_cache_dir", "cache/vlrm_outputs")
    stage2_processor = Stage2_VLRM_CLIP_Fusion(device=device, cache_dir=vlrm_cache_dir)
    
    results = []
    
    for s2_idx, s1_out in enumerate(tqdm(stage1_results, desc="Stage 2")):
        fname = _fix_fname(s1_out["file_name"])
        image_id = s1_out.get("image_id", os.path.basename(fname).split('.')[0])
        
        # Reconstruct GDINO boxes 
        gdino_boxes = None
        if len(s1_out["gdino_boxes"]) > 0:
             gdino_boxes = Boxes(s1_out["gdino_boxes"]) 
        
        # MaskRCNN boxes
        m_boxes = s1_out["maskrcnn"]["boxes"]
        m_scores = s1_out["maskrcnn"]["scores"]
        m_classes = s1_out["maskrcnn"]["classes"]
        
        labeled_boxes = torch.empty(0, 4, device=device)
        labeled_scores = torch.empty(0, device=device)
        labeled_classes = torch.empty(0, dtype=torch.int64, device=device)

        stage2_threshold = param_dict.get("stage2_threshold", 0.20)
        if s2_idx == 0:
            print(f"[Stage 2] Using threshold: {stage2_threshold} for COCO ({len(coco_classes)} classes)")

        if len(m_boxes) > 0:
            bg_boxes, _, _ = extract_background_proposals(
                gdino_boxes, m_boxes, m_scores, m_classes, iou_threshold=0.5
            )
            
            if len(bg_boxes) > 0:
                try:
                    pil_image = Image.open(fname).convert("RGB")
                    
                    semantic_scores, valid_indices = stage2_processor.run_stage_2(
                        background_rois=bg_boxes.to(device), 
                        full_image=pil_image,
                        class_names=coco_classes,
                        image_id=str(image_id),
                        image_path=fname,
                        force_vlrm_rerun=force_vlrm_rerun,
                        verbose=verbose,
                        tracker=tracker
                    )
                    
                    if semantic_scores.numel() > 0 and len(valid_indices) > 0:
                        conf, cls_idx = torch.max(semantic_scores, dim=1)
                        # Use threshold from param_dict (default 0.20 for COCO's 65 classes)
                        stage2_threshold = param_dict.get("stage2_threshold", 0.20)
                        mask = conf > stage2_threshold
                        
                        keep_local = torch.nonzero(mask).squeeze(1)
                        # Safeguard against list index out of range if shapes mismatch
                        keep_local = keep_local[keep_local < len(valid_indices)]
                        if keep_local.numel() > 0:
                            orig_indices = [valid_indices[k] for k in keep_local.tolist()]
                            labeled_boxes = bg_boxes[orig_indices]
                            labeled_scores = conf[keep_local].cpu()
                            labeled_classes = cls_idx[keep_local].cpu()
                
                except Exception as e:
                    print(f"Error in Stage 2 for {fname}: {e}")
        
        res_item = {
            "file_name": fname,
            "image_id": image_id,
            "stage2_boxes": labeled_boxes,
            "stage2_scores": labeled_scores,
            "stage2_classes": labeled_classes
        }
        results.append(res_item)
        
    del stage2_processor
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after_stage2_coco")
            
    return results

def run_stage3_coco(stage1_results, stage2_results, param_dict, evaluator):
    """
    Run Stage 3: Merge, OT Fusion, SAM, SRM, NMS.
    Evaluates results using the provided evaluator.
    """
    print("[COCO Stage 3] Merging and Evaluating...")
    
    sam = param_dict.get("sam")
    debug_mode = param_dict.get("debug_mode", True)
    tracker = param_dict.get("tracker")
    if tracker is None:
        verbose_level = param_dict.get("verbose_level", "medium")
        out_dir = param_dict.get("out_dir", "debug_outputs")
        tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, "debug_outputs"))
        param_dict["tracker"] = tracker

    # Initialize SRM only if enabled in params
    use_srm = param_dict.get("use_srm", True)
    srm = ScoreRefinementModule(per_image_norm=True) if use_srm else None
    if use_srm:
        print("[SRM] Enabled for COCO Stage 3")
    resize_transform = param_dict.get("resize_transform")
    verbose = param_dict.get("verbose", False)
    device = param_dict.get("device", "cuda")
    
    if len(stage1_results) != len(stage2_results):
        print(f"Error: Mismatch S1 vs S2 results.")
        return {}
    
    evaluator.reset()

    is_sam3 = (sam is not None and sam.__class__.__name__ == "SAM3CallableWrapper")
    cache_subdir = "sam3_cache" if is_sam3 else "sam_cache"
    sam_cache_dir = os.path.join(param_dict.get("out_dir", "outputs_coco"), cache_subdir)
    existing_cached = []
    if sam_cache_dir and os.path.exists(sam_cache_dir):
        existing_cached = [f for f in os.listdir(sam_cache_dir) if f.startswith('final_')]
        print(f"[Resume] Found {len(existing_cached)} cached results from previous run")

    skipped_count = 0

    for i in tqdm(range(len(stage1_results)), desc="Stage 3"):
        s1 = stage1_results[i]
        s2 = stage2_results[i]

        fname = _fix_fname(s1["file_name"])
        height = s1["height"]
        width = s1["width"]
        image_id = s1.get("image_id")

        cached_result_path = None
        if sam_cache_dir:
            safe_id = str(image_id).replace('/', '_').replace('\\', '_') if image_id else os.path.basename(fname).split('.')[0]
            cached_result_path = os.path.join(sam_cache_dir, f"final_{safe_id}.pkl")

        if cached_result_path and os.path.exists(cached_result_path):
            try:
                cached = torch.load(cached_result_path)
                # Validate content hash: reject stale cache from prior runs
                # with different Stage 2 outputs or hyperparameters.
                s2_hash = hashlib.md5(
                    s2["stage2_boxes"].cpu().numpy().tobytes() +
                    s2["stage2_scores"].cpu().numpy().tobytes()
                ).hexdigest()[:12]
                cached_hash = cached.get('s2_hash', None)
                if cached_hash is not None and cached_hash != s2_hash:
                    print(f"[Resume] STALE cache for {image_id} (s2 changed) — re-processing")
                    try:
                        os.remove(cached_result_path)
                    except OSError:
                        pass
                else:
                    inst = Instances((height, width))
                    inst.pred_boxes = Boxes(cached['boxes'])
                    inst.scores = cached['scores']
                    inst.pred_classes = cached['classes']
                    inputs_mock = [{"file_name": fname, "height": height, "width": width, "image_id": image_id}]
                    outputs = [{"instances": inst}]
                    evaluator.process(inputs_mock, outputs)
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"[Resume] Failed to load cached result for {image_id}: {e}. Deleting corrupted file.")
                try:
                    os.remove(cached_result_path)
                except OSError:
                    pass
        
        g_boxes = s1["gdino_boxes"]
        g_scores = s1["gdino_scores"]
        g_classes = s1["gdino_classes"]
        
        s2_boxes = s2["stage2_boxes"]
        s2_scores = s2["stage2_scores"]
        s2_classes = s2["stage2_classes"]

        # Extract known boxes from Stage 1 Mask-RCNN output
        known_boxes = s1["maskrcnn"].get("known_boxes", torch.empty(0, 4, device=device))
        known_scores = s1["maskrcnn"].get("known_scores", torch.empty(0, device=device))
        known_classes = s1["maskrcnn"].get("known_classes", torch.empty(0, dtype=torch.int64, device=device))
        
        t_start_ot = tracker.log_module_start("Hungarian-OT-Fusion") if tracker else None
        merged_boxes, merged_scores, merged_classes = ot_fusion(
            s2_boxes.to(device), s2_scores.to(device), s2_classes.to(device),
            known_boxes.to(device), known_scores.to(device), known_classes.to(device),
            g_boxes.to(device), g_scores.to(device), g_classes.to(device),
            iou_weight=0.6, semantic_weight=0.4, sinkhorn_reg=0.1, sinkhorn_iters=50,
            param_dict={'verbose': verbose}
        )
        if tracker:
            tracker.log_module_end("Hungarian-OT-Fusion", t_start_ot, metadata={
                "pair_count": len(merged_boxes),
                "cost_summary": {"total_cost": 0} # Placeholder
            })

        final_boxes, final_scores, final_classes = _apply_sam_srm_nms(
            merged_boxes, merged_scores, merged_classes,
            fname, sam, srm, resize_transform, verbose, device, tracker=tracker,
            debug_mode=param_dict.get("debug_mode"), sam_cache_dir=sam_cache_dir
        )
        
        if tracker and debug_mode and len(final_boxes) > 0:
            tracker.save_debug_image("Final-Detection", fname, boxes=final_boxes[:10])
        
        inst = Instances((height, width))
        inst.pred_boxes = Boxes(final_boxes)
        inst.scores = final_scores
        inst.pred_classes = final_classes

        if cached_result_path:
            temp_path = cached_result_path + ".tmp"
            # Compute content hash from Stage 2 inputs for cache validation
            s2_hash = hashlib.md5(
                s2["stage2_boxes"].cpu().numpy().tobytes() +
                s2["stage2_scores"].cpu().numpy().tobytes()
            ).hexdigest()[:12]
            try:
                os.makedirs(sam_cache_dir, exist_ok=True)
                torch.save({
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'classes': final_classes,
                    'image_id': image_id,
                    's2_hash': s2_hash
                }, temp_path)
                os.replace(temp_path, cached_result_path)
            except Exception as e:
                print(f"[Resume] Failed to save result: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        inputs_mock = [{
            "file_name": fname,
            "height": height,
            "width": width,
            "image_id": image_id
        }]

        outputs = [{"instances": inst}]

        evaluator.process(inputs_mock, outputs)

    print(f"[Resume] Stage 3 complete: {skipped_count} skipped (cached), {len(stage1_results)-skipped_count} processed")
    return evaluator.evaluate()

def _get_sam_cache_path(image_path, image_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    safe_id = str(image_id).replace('/', '_').replace('\\', '_')
    return os.path.join(cache_dir, f"sam_{safe_id}.pkl")

def _apply_sam_srm_nms(boxes, scores, classes, image_path, sam, srm, resize_transform, verbose, device, tracker=None, debug_mode=False, sam_cache_dir=None):
    """
    Applies SAM mask quality scoring, SRM refinement, and NMS.
    """
    if len(boxes) == 0:
        return boxes, scores, classes

    # Ensure correct device and float types
    boxes = boxes.float().to(device)
    scores = scores.float().to(device)
    classes = classes.long().to(device)

    # 1. Pre-SAM NMS (Change 6)
    if len(boxes) > 0:
        keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

    # 2. Pre-SAM Top-K cap (Change 6)
    max_before_sam = 300
    if len(scores) > max_before_sam:
        topk_scores, topk_idxs = torch.topk(scores, max_before_sam)
        boxes = boxes[topk_idxs]
        scores = topk_scores
        classes = classes[topk_idxs]

    sam_scores = None
    sam_refined = None
    image_id = os.path.basename(image_path).split('.')[0]

    cache_available = sam_cache_dir is not None
    cache_path = _get_sam_cache_path(image_path, image_id, sam_cache_dir) if cache_available else None

    if cache_available and cache_path and os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path)
            cached_sam_scores = cached['sam_scores'].to(device)
            cached_sam_refined = cached['sam_refined'].to(device)
            cached_hash = cached.get('boxes_hash', None)

            # Validate both shape AND content hash (matching LVIS pattern).
            boxes_hash_check = hashlib.md5(boxes.cpu().numpy().tobytes()).hexdigest()[:12]
            if len(cached_sam_refined) == len(scores) and (cached_hash is None or cached_hash == boxes_hash_check):
                sam_scores = cached_sam_scores
                sam_refined = cached_sam_refined
                boxes = sam_refined
                print(f"[SAM Cache] Loaded: {os.path.basename(cache_path)}")
            else:
                reason = "count mismatch" if len(cached_sam_refined) != len(scores) else "content changed"
                print(f"[SAM Cache] STALE ({reason}): {os.path.basename(cache_path)} — re-running SAM")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[SAM Cache] Load failed for {image_id}: {e}. Deleting corrupted file.")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    if sam is not None and sam_refined is None:
        try:
            if resize_transform is None:
                resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

            curr_image = cv2.imread(image_path)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            img_shape = curr_image.shape[:2]

            curr_image = resize_transform.apply_image(curr_image)
            curr_image = torch.as_tensor(curr_image, device=sam.device).permute(2, 0, 1).contiguous()

            boxes_device = boxes.to(sam.device)
            sam_box_prompts = resize_transform.apply_boxes_torch(boxes_device, img_shape)

            batch_size = 50
            sam_scores_list = []
            sam_refined_list = []

            for k in range(0, len(sam_box_prompts), batch_size):
                batch = sam_box_prompts[k:k+batch_size]
                if len(batch) == 0:
                    continue

                inp = [{
                    "image": curr_image,
                    "boxes": batch,
                    "original_size": img_shape
                }]

                if tracker:
                    t_start_sam = tracker.log_module_start("SAM")

                out = sam(inp, multimask_output=False)

                masks = out[0]['masks'].clone().detach()
                sam_refined_list.append(batched_mask_to_box(masks).squeeze(1))
                sam_scores_list.append(out[0]['iou_predictions'])

                if tracker:
                    tracker.log_module_end("SAM", t_start_sam, metadata={
                        "num_masks": len(masks),
                        "coverage": masks.float().mean().item() * 100
                    })
                    if debug_mode:
                        tracker.save_debug_image("SAM", image_path, boxes=boxes[k:k+batch_size][:10], masks=masks[:10])

                del out, masks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if len(sam_scores_list) > 0:
                sam_scores = torch.cat(sam_scores_list, dim=0).squeeze(1)
                sam_refined = torch.cat(sam_refined_list, dim=0)

                boxes = sam_refined

                if cache_available and cache_path and len(sam_scores) > 0:
                    temp_path = cache_path + ".tmp"
                    boxes_hash_save = hashlib.md5(boxes.cpu().numpy().tobytes()).hexdigest()[:12]
                    try:
                        os.makedirs(sam_cache_dir, exist_ok=True)
                        torch.save({
                            'sam_scores': sam_scores,
                            'sam_refined': sam_refined,
                            'boxes_count': len(boxes),
                            'boxes_hash': boxes_hash_save,
                            'image_id': image_id
                        }, temp_path)
                        os.replace(temp_path, cache_path)
                        print(f"[SAM Cache] Saved: {os.path.basename(cache_path)}")
                    except Exception as e:
                        print(f"[SAM Cache] Save failed: {e}")
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass

        except Exception as e:
            if verbose:
                print(f"[SAM] Warning: {e}")
            sam_scores = None

    if srm is not None and sam_scores is not None and len(sam_scores) == len(scores):
        try:
            if tracker:
                t_start_srm = tracker.log_module_start("SRM")
            
            orig_mean = scores.mean().item()
            scores = srm.refine_scores(scores, sam_scores)
            new_mean = scores.mean().item()
            
            if tracker:
                tracker.log_module_end("SRM", t_start_srm, metadata={
                    "score_delta": {
                        "before_mean": orig_mean,
                        "after_mean": new_mean,
                        "change_percent": (new_mean - orig_mean) / (orig_mean + 1e-6) * 100
                    }
                })
        except Exception as e:
            if verbose: print(f"[SRM] Warning: {e}")

    boxes = boxes.float()
    scores = scores.float()
    classes = classes.long()
    
    # GPU: NMS on device
    keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
    boxes = boxes[keep]
    scores = scores[keep]
    classes = classes[keep]
    
    if len(scores) > 100:
        topk_scores, topk_idxs = torch.topk(scores, 100)
        boxes = boxes[topk_idxs]
        scores = topk_scores
        classes = classes[topk_idxs]
    
    # Move to CPU for evaluator compatibility
    return boxes.cpu(), scores.cpu(), classes.cpu()
