import numpy as np
import transforms as T2
import torch
import cv2
import open_clip
import torch.nn.functional as F
import sys
import os

# Add parent scripts directory to path for SRM module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sklearn.preprocessing import MinMaxScaler
from srm_module import ScoreRefinementModule
from utils import BBoxVisualizer
from PIL import Image
from detectron2.data import MetadataCatalog
from torchvision.ops import box_convert
from torchvision.ops.boxes import batched_nms
from detectron2.structures import Instances, Boxes, pairwise_iou
from pathlib import Path
from torch import nn
from segment_anything.utils.amg import batched_mask_to_box

# ==============================================================================
# Stage 2: Caption Blacklist (generic/scene descriptions to reject)
# ==============================================================================
CAPTION_BLACKLIST = {"background", "image", "picture", "photo", "scene", "blurry", "blur", 
                     "photograph", "abstract", "wall", "row", "ocean", "sky", "floor", "ground"}

def prepare_image_for_GDINO(input, device = "cuda"):
    """
    inputs: dict, with keys "file_name", "height", "width", "image", "image_id"
    outputs: transformed images
    """
    transform = T2.Compose(
        [
            T2.RandomResize([800], max_size=1333),
            T2.ToTensor(),
            T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    image_src = Image.open(input["file_name"]).convert("RGB")
    image = np.asarray(image_src)
    image_transformed, _ = transform(image_src, None)
    image_transformed = image_transformed.to(device)
    return image_transformed[None], image


def extract_rpn_proposals(torchvision_maskrcnn, image_path, device):
    """
    Extract RPN proposals from TorchVision Mask-RCNN using forward hook.
    
    Model device is managed externally (GPU during Stage 1, CPU after Stage 1).
    No per-image CPU/GPU moves — stays on its current device for the whole batch.
    
    Returns:
        boxes: Tensor (N, 4) in xyxy format, on `device`
        scores: Tensor (N,) objectness scores, on `device`
        h, w: original image dimensions
    """
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    
    # Use model's current device (GPU during Stage 1)
    model_device = next(torchvision_maskrcnn.parameters()).device
    

    # Hook to capture RPN proposals
    rpn_boxes = []
    rpn_scores = []
    
    def capture_proposals(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            boxes_list, scores_list = output[0], output[1]
            if isinstance(boxes_list, list) and len(boxes_list) > 0:
                rpn_boxes.append(boxes_list[0])
                if isinstance(scores_list, list) and len(scores_list) > 0:
                    rpn_scores.append(scores_list[0])
    
    hook = torchvision_maskrcnn.rpn.register_forward_hook(capture_proposals)
    
    try:
        with torch.no_grad():
            torchvision_maskrcnn.eval()
            _ = torchvision_maskrcnn([image_tensor.to(model_device)])
    finally:
        hook.remove()
    

    # Process captured proposals
    if rpn_boxes and len(rpn_boxes[0]) > 0:
        boxes = rpn_boxes[0]
        # Scale proposals back to the original image size because the hook captures them
        # at the model's internal resized scale.
        try:
            with torch.no_grad():
                images, _ = torchvision_maskrcnn.transform([image_tensor.to(model_device)])
                resized_h, resized_w = images.image_sizes[0]
                scale_x = w / resized_w
                scale_y = h / resized_h
                
                # Clone to avoid inplace modification of captured RPN tensors
                boxes = boxes.clone()
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
        except Exception as scale_err:
            print(f"[Mask-RCNN Warning] Failed to scale RPN proposals: {scale_err}")

        scores = rpn_scores[0] if rpn_scores else torch.ones(len(boxes), device=model_device)
    else:
        # Fallback to final detections
        with torch.no_grad():
            outputs = torchvision_maskrcnn([image_tensor.to(model_device)])
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
    
    # Return on the caller's target device
    return boxes.to(device), scores.to(device), h, w

@torch.no_grad()
def run_stage1_inference(inputs, param_dict):
    """
    Stage 1: Initialization (Mask R-CNN + GroundingDINO)
    
    Returns:
        dict: {
            "file_name": str,
            "height": int,
            "width": int,
            "known": dict (boxes, scores, labels),
            "background": dict (boxes, objectness),
            "gdino": dict (boxes, scores, labels),
        }
    """
    rcnn_model = param_dict["rcnn_model"]
    # Provide backward compatibility for GDINO model if passed as arg in param_dict
    model = param_dict.get("gdino_model") 
    
    torchvision_maskrcnn = param_dict.get("torchvision_maskrcnn")
    device = param_dict["device"]
    coco_to_lvis = param_dict["coco_to_lvis"]
    
    # 1. Mask R-CNN (Known Classes)
    rcnn_model.eval()
    
    # Configure NMS/Score thresholds
    if not isinstance(rcnn_model.roi_heads.box_predictor, nn.ModuleList):
        rcnn_model.proposal_generator.nms_thresh = 0.9
    else:
        rcnn_model.proposal_generator.nms_thresh_train = 0.9
        rcnn_model.proposal_generator.nms_thresh_test = 0.9

    if isinstance(rcnn_model.roi_heads.box_predictor, nn.ModuleList):
        box_predictors = rcnn_model.roi_heads.box_predictor
    else:
        box_predictors = [rcnn_model.roi_heads.box_predictor]

    for box_predictor in box_predictors:
        box_predictor.allow_novel_classes_during_inference = True
        box_predictor.test_topk_per_image = 300
        box_predictor.test_nms_thresh = 0.5
        box_predictor.test_score_thresh = 0.0001

    # Use autocast for Mask-RCNN forward pass
    with torch.cuda.amp.autocast():
        outputs = rcnn_model(inputs)
        
    rcnn_boxes = outputs[0]["instances"].pred_boxes.tensor
    rcnn_scores = outputs[0]["instances"].scores
    rcnn_classes = outputs[0]["instances"].pred_classes

    # Filter Known Classes (COCO < 80)
    known_mask = rcnn_classes < 80
    known_boxes = rcnn_boxes[known_mask]
    known_scores = rcnn_scores[known_mask]
    known_classes = rcnn_classes[known_mask]
    known_classes_lvis = torch.tensor([coco_to_lvis[c.item()] for c in known_classes], device=device)

    # 2. Mask R-CNN (RPN / Background Proposals)
    bg_boxes = torch.empty(0, 4, device=device)
    bg_objectness = torch.empty(0, device=device)
    
    if torchvision_maskrcnn is not None:
        image_path = inputs[0]['file_name']
        bg_boxes, bg_objectness, _, _ = extract_rpn_proposals(
            torchvision_maskrcnn, image_path, device
        )
        
        # GPU Optimization: Reduce proposals
        if len(bg_boxes) > 500:
            top_k = 500
            top_indices = torch.topk(bg_objectness, min(top_k, len(bg_objectness)))[1]
            bg_boxes = bg_boxes[top_indices]
            bg_objectness = bg_objectness[top_indices]
    else:
        # Fallback
        bg_boxes_idxs = rcnn_classes == 80
        bg_boxes = rcnn_boxes[bg_boxes_idxs]
        bg_objectness = rcnn_scores[bg_boxes_idxs]

    # 3. GroundingDINO
    gdino_image, _ = prepare_image_for_GDINO(inputs[0], device=device)
    
    text_prompt_list = param_dict["text_prompt_list"]
    positive_map_list = param_dict["positive_map_list"]
    length = param_dict["class_len_per_prompt"]
    
    all_out_logits = []
    all_out_bbox = []
    all_prob_to_token = []

    # Process GDINO in batches of 3 captions to avoid OOM with 16 GB VRAM
    GDINO_BATCH_SIZE = 9
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i in range(0, len(text_prompt_list), GDINO_BATCH_SIZE):
            batch_captions = text_prompt_list[i:i+GDINO_BATCH_SIZE]
            batched_image = gdino_image.repeat(len(batch_captions), 1, 1, 1)
            curr_output = model(batched_image, captions=batch_captions)
            all_out_logits.append(curr_output["pred_logits"])
            all_out_bbox.append(curr_output["pred_boxes"])
            all_prob_to_token.append(curr_output["pred_logits"].sigmoid())

    prob_to_token = torch.cat(all_prob_to_token, dim=0)
    out_bbox = torch.cat(all_out_bbox, dim=0)

    prob_to_label_list = []
    for i in range(prob_to_token.shape[0]):
        curr_prob_to_label = prob_to_token[i] @ positive_map_list[i].to(prob_to_token.device).T
        prob_to_label_list.append(curr_prob_to_label)

    prob_to_label = torch.cat(prob_to_label_list, dim = 1)
    
    # Top-k selection
    topk_values, topk_idxs = torch.topk(prob_to_label.view(-1), 900, 0)
    gdino_scores = topk_values
    
    # Logic for box indices
    labels = topk_idxs % prob_to_label.shape[1]
    topk_boxes = topk_idxs // prob_to_label.shape[1]
    topk_boxes_batch_idx = labels // length # This is 0 for batch size 1 usually
    
    combined_box_index = torch.stack((topk_boxes_batch_idx, topk_boxes), dim=1)
    
    gdino_boxes_norm = out_bbox[combined_box_index[:, 0], combined_box_index[:, 1]]
    
    h, w = inputs[0]['height'], inputs[0]['width']
    gdino_boxes = gdino_boxes_norm * torch.tensor([w, h, w, h], device=gdino_boxes_norm.device)
    gdino_boxes = box_convert(boxes=gdino_boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # Filter by score threshold
    SCORE_THRESHOLD = 0.05
    mask = gdino_scores >= SCORE_THRESHOLD
    gdino_scores = gdino_scores[mask]
    gdino_boxes = gdino_boxes[mask]
    gdino_labels = labels[mask]

    # =========================================================================
    # Stage 1 GDINO Post-Processing: NMS only.
    # SAM refinement is DELAYED to Stage 3 (per architecture spec).
    # When param_dict["sam"] = None (set by main.py for Stage 1), this block
    # is skipped and raw NMS-filtered GDINO boxes are passed to Stage 3.
    # Stage 3 then runs SAM on ALL candidate boxes (known + VLRM + GDINO).
    # =========================================================================
    if len(gdino_boxes) > 0:
        gdino_boxes = gdino_boxes.to(device)
        gdino_scores = gdino_scores.to(device)
        gdino_labels = gdino_labels.to(device)
        keep = batched_nms(gdino_boxes, gdino_scores, gdino_labels, iou_threshold=0.5)
        gdino_boxes = gdino_boxes[keep]
        gdino_scores = gdino_scores[keep]
        gdino_labels = gdino_labels[keep]

    sam = param_dict.get("sam")
    resize_transform = param_dict.get("resize_transform")

    if sam is not None and len(gdino_boxes) > 0:
        # SAM in Stage 1 path (only used if sam is explicitly passed — NOT in LVIS pipeline)
        gdino_boxes = gdino_boxes.to(sam.device)
        curr_image = cv2.imread(inputs[0]['file_name'])
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        img_shape = curr_image.shape[:2]
        if resize_transform is None:
            from segment_anything.utils.transforms import ResizeLongestSide
            resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
        
        curr_image_resized = resize_transform.apply_image(curr_image)
        curr_image_tensor = torch.as_tensor(curr_image_resized, device=sam.device).permute(2, 0, 1).contiguous()

        sam_box_prompts = resize_transform.apply_boxes_torch(gdino_boxes, img_shape)

        sam_batch_size = 50
        sam_refined_boxes_list = []
        sam_scores_list = []

        for i in range(0, len(sam_box_prompts), sam_batch_size):
            batch_sam_box_prompts = sam_box_prompts[i:i + sam_batch_size]
            if len(batch_sam_box_prompts) == 0:
                continue

            batched_input = [{
                "image": curr_image_tensor,
                "boxes": batch_sam_box_prompts,
                "original_size": img_shape,
            }]

            batched_output = sam(batched_input, multimask_output=False)
            sam_refined_boxes_list.append(batched_mask_to_box(batched_output[0]['masks'].clone().detach()).squeeze(1))
            sam_scores_list.append(batched_output[0]['iou_predictions'])

            del batched_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(sam_refined_boxes_list) > 0:
            sam_refined_boxes = torch.cat(sam_refined_boxes_list, dim=0)
            sam_scores = torch.cat(sam_scores_list, dim=0).squeeze(1)

            if len(sam_scores) > 0:
                print(f"[SRM] Refining {len(gdino_scores)} scores using SAM mask quality...")
                srm = ScoreRefinementModule(per_image_norm=True)
                gdino_scores = srm.refine_scores(gdino_scores, sam_scores)
                print(f"[SRM] ✓ Scores refined: range=[{gdino_scores.min().item():.3f}, {gdino_scores.max().item():.3f}]")

            # Keep top 100 when SAM runs in Stage 1
            if len(gdino_scores) > 100:
                topk_scores, topk_idxs = torch.topk(gdino_scores, 100)
                gdino_boxes = sam_refined_boxes[topk_idxs]
                gdino_labels = gdino_labels[topk_idxs]
                gdino_scores = topk_scores
            else:
                gdino_boxes = sam_refined_boxes
    else:
        # Delayed SAM mode (LVIS pipeline): SAM=None in Stage 1.
        # Apply a soft cap of 300 raw GDINO boxes to prevent Stage 3 memory explosion.
        # These raw boxes will be refined by SAM in Stage 3 alongside known+VLRM boxes.
        MAX_RAW_GDINO = 300
        if len(gdino_scores) > MAX_RAW_GDINO:
            topk_scores, topk_idxs = torch.topk(gdino_scores, MAX_RAW_GDINO)
            gdino_boxes = gdino_boxes[topk_idxs]
            gdino_labels = gdino_labels[topk_idxs]
            gdino_scores = topk_scores

    return {
        "file_name": inputs[0]['file_name'],
        "image_id": inputs[0]['image_id'],
        "height": h,
        "width": w,
        "known": {
            "boxes": known_boxes,
            "scores": known_scores,
            "labels": known_classes_lvis
        },
        "background": {
            "boxes": bg_boxes,
            "objectness": bg_objectness
        },
        "gdino": {
            "boxes": gdino_boxes,
            "scores": gdino_scores,
            "labels": gdino_labels
        }
    }


@torch.no_grad()
def run_stage3_inference(stage1_out, stage2_out_bg, param_dict):
    """
    Stage 3: Refinement (OT Fusion + NMS + SAM + SRM)
    
    Args:
        stage1_out: Output from run_stage1_inference
        stage2_out_bg: Dict { "boxes": Tensor, "scores": Tensor, "labels": Tensor } 
                       (Labeled Background Boxes from Stage 2)
    """
    sam = param_dict["sam"]
    resize_transform = param_dict["resize_transform"]
    visualize = param_dict.get("visualize", False)
    out_dir = param_dict["out_dir"]
    lvis_data_split = param_dict["lvis_data_split"]
    device = param_dict.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Unpack Stage 1 — ensure all on GPU
    known_boxes = stage1_out["known"]["boxes"].to(device)
    known_scores = stage1_out["known"]["scores"].to(device)
    known_classes = stage1_out["known"]["labels"].to(device)
    
    gdino_boxes = stage1_out["gdino"]["boxes"].to(device)
    gdino_scores = stage1_out["gdino"]["scores"].to(device)
    gdino_labels = stage1_out["gdino"]["labels"].to(device)
    
    # Unpack Stage 2 (VLRM Results) — ensure on GPU
    if stage2_out_bg is not None and len(stage2_out_bg["boxes"]) > 0:
        vlrm_bg_boxes = stage2_out_bg["boxes"].to(device)
        vlrm_bg_scores = stage2_out_bg["scores"].to(device)
        vlrm_bg_classes = stage2_out_bg["labels"].to(device)
    else:
        vlrm_bg_boxes = torch.empty(0, 4, device=device)
        vlrm_bg_scores = torch.empty(0, device=device)
        vlrm_bg_classes = torch.empty(0, device=device)
    
    # Adaptive Proposal Fusion (WBF) — replaces OT Fusion + NMS
    use_apf = param_dict.get("use_apf", True)
    if use_apf:
        apf = AdaptiveProposalFusion(
            source_weights=param_dict.get("apf_source_weights", {0: 0.55, 1: 0.50, 2: 0.40}),
            wbf_iou_threshold=param_dict.get("apf_iou_threshold", 0.5),
            wbf_score_threshold=param_dict.get("apf_score_threshold", 0.05),
            wbf_max_boxes=param_dict.get("apf_max_boxes", 500),
            device=device,
        )
        proposals_by_source = {
            0: {"boxes": known_boxes,  "scores": known_scores,  "classes": known_classes},
            1: {"boxes": gdino_boxes,  "scores": gdino_scores,  "classes": gdino_labels},
            2: {"boxes": vlrm_bg_boxes, "scores": vlrm_bg_scores, "classes": vlrm_bg_classes},
        }
        apf_out = apf.fuse(proposals_by_source)
        boxes = apf_out["boxes"]
        scores = apf_out["scores"]
        labels = apf_out["classes"]
        source_contrib = apf_out["source_contributions"]
    else:
        # Fallback: original OT Fusion
        combined_rcnn_boxes = torch.cat([known_boxes, vlrm_bg_boxes], dim=0)
        combined_rcnn_scores = torch.cat([known_scores, vlrm_bg_scores], dim=0)
        combined_rcnn_classes = torch.cat([known_classes, vlrm_bg_classes], dim=0)

        from ot_fusion_module import ot_fusion
        boxes, scores, labels = ot_fusion(
            gdino_boxes, gdino_scores, gdino_labels,
            combined_rcnn_boxes, combined_rcnn_scores, combined_rcnn_classes,
            iou_weight=0.6, semantic_weight=0.4, sinkhorn_reg=0.1, sinkhorn_iters=50,
            boost_factor=1.5, penalty_factor=0.7,
            param_dict=param_dict
        )
        labels = labels.to(torch.int64)

        if len(boxes) > 0:
            boxes = boxes.float().to(device)
            scores = scores.float().to(device)
            labels = labels.to(device)
            keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]
        source_contrib = torch.empty(0, 3, device=device)

    pre_nms_total = len(known_boxes) + len(gdino_boxes) + len(vlrm_bg_boxes)

    if len(boxes) == 0:
        result = Instances((stage1_out["height"], stage1_out["width"]))
        result.pred_boxes = Boxes(torch.empty(0, 4, device=device))
        result.scores = torch.empty(0, device=device)
        result.pred_classes = torch.empty(0, dtype=torch.int64, device=device)
        return [{"instances": result}]

    # Top-K pre-filter before SAM (cap at 500 for LVIS)
    max_before_sam = 500
    if len(scores) > max_before_sam:
        topk_scores, topk_idxs = torch.topk(scores, max_before_sam)
        boxes = boxes[topk_idxs]
        scores = topk_scores
        labels = labels[topk_idxs]

    src_summary = ""
    if use_apf and len(source_contrib) > 0:
        src_summary = f", src_contrib=[{source_contrib.mean(dim=0)[0]:.2f},{source_contrib.mean(dim=0)[1]:.2f},{source_contrib.mean(dim=0)[2]:.2f}]"
    print(f"[Merge] GDINO: {len(gdino_boxes)}, RCNN: {len(known_boxes)}, VLRM: {len(vlrm_bg_boxes)} "
          f"→ Pre-APF: {pre_nms_total} → After APF+NMS: {len(boxes)}{src_summary}")

    # SAM Refinement
    boxes = boxes.to(sam.device)
    curr_image = cv2.imread(stage1_out["file_name"])
    curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
    img_shape = curr_image.shape[:2] #(h, w)
    
    # Scale image for SAM
    curr_image_tensor = resize_transform.apply_image(curr_image)
    curr_image_tensor = torch.as_tensor(curr_image_tensor, device=sam.device).permute(2, 0, 1).contiguous()
    
    sam_box_prompts = resize_transform.apply_boxes_torch(boxes, img_shape)
    
    sam_masks_list = []
    sam_refined_boxes_list = []
    sam_scores_list = []
    
    sam_batch_size = 64  # Increased from 50 for better GPU utilization
    for i in range(0, len(sam_box_prompts), sam_batch_size):
        batch_prompts = sam_box_prompts[i:i+sam_batch_size]
        if len(batch_prompts) == 0: continue
        
        batched_input = [{
            "image": curr_image_tensor,
            "boxes": batch_prompts,
            "original_size": img_shape
        }]
        batched_output = sam(batched_input, multimask_output=False)
        sam_masks_list.append(batched_output[0]['masks'])
        sam_refined_boxes_list.append(batched_mask_to_box(batched_output[0]['masks'].clone().detach()).squeeze(1))
        sam_scores_list.append(batched_output[0]['iou_predictions'])
        
    if len(sam_masks_list) > 0:
        sam_refined_boxes = torch.cat(sam_refined_boxes_list, dim=0)
        sam_scores = torch.cat(sam_scores_list, dim=0)
    else:
        sam_refined_boxes = torch.empty(0, 4, device=device)
        sam_scores = torch.empty(0, device=device)
        
    # GPU: Keep SAM scores on GPU for SRM (pure-torch MinMaxScaler)
    sam_scores = sam_scores.squeeze(1)

    # SRM Refinement - enable if use_srm is True (default enabled)
    use_srm = param_dict.get("use_srm", True)
    if use_srm and len(sam_scores) > 0 and len(scores) > 0:
        min_len = min(len(scores), len(sam_scores))
        scores = scores[:min_len].to(device)
        sam_scores = sam_scores[:min_len].to(device)
        labels = labels[:min_len]
        sam_refined_boxes = sam_refined_boxes[:min_len]

        srm = ScoreRefinementModule(per_image_norm=True)
        scores = srm.refine_scores(scores, sam_scores)
        print(f"[SRM] Enabled - refined {len(scores)} detections")
        
    # Final Top-K Selection (on GPU, move to CPU only for evaluator output)
    k = min(500, len(scores))
    if k > 0:
        topk_scores, topk_idxs = torch.topk(scores, k)
        topk_idxs = topk_idxs.to(torch.int64)
        boxes = sam_refined_boxes[topk_idxs]
        labels = labels[topk_idxs]
        scores = topk_scores
    else:
        boxes = torch.empty(0, 4, device=device)
        labels = torch.empty(0, dtype=torch.int64, device=device)
        scores = torch.empty(0, device=device)
        
    # Visualization
    if visualize:
        result = Instances((stage1_out["height"], stage1_out["width"]))
        result.pred_boxes = Boxes(boxes[:5])
        result.scores = scores[:5]
        result.pred_classes = labels[:5]
        
        meta_data = MetadataCatalog.get(lvis_data_split)
        Path(f"{out_dir}/output_images").mkdir(parents=True, exist_ok=True)
        
        # Draw on original image
        v = BBoxVisualizer(curr_image, meta_data, scale=1.2)
        out = v.draw_instance_predictions(result)
        cv2.imwrite(f"{out_dir}/output_images/{stage1_out['file_name'].split('/')[-1]}", out.get_image()[:, :, ::-1])

    # Format Output for Evaluator
    # Move to CPU for detectron2 evaluator compatibility
    result = Instances((stage1_out["height"], stage1_out["width"]))
    result.pred_boxes = Boxes(boxes.cpu())
    result.scores = scores.cpu()
    result.pred_classes = labels.cpu()
    
    final_outputs = []
    curr_output = {}
    curr_output['instances'] = result
    final_outputs.append(curr_output)
    
    return final_outputs


@torch.no_grad()
def inference_gdino(model, inputs, text_prompt_list, param_dict):
    """
    Legacy wrapper for backward compatibility.
    Runs full pipeline effectively by calling Stage 1 -> Inline Stage 2 logic -> Stage 3.
    """
    device = param_dict.get("device", "cuda")
    
    # Backward compatibility wrapper
    param_dict["gdino_model"] = model
    param_dict["text_prompt_list"] = text_prompt_list
    
    # 1. Run Stage 1
    s1_out = run_stage1_inference(inputs, param_dict)
    
    # 2. Run Stage 2 Logic (Inline)
    bg_boxes = s1_out["background"]["boxes"]
    
    s2_bg_boxes = []
    s2_bg_scores = []
    s2_bg_labels = []
    
    if len(bg_boxes) > 0 and param_dict.get("vlrm_model"):
        vlrm = param_dict.get('vlrm_model')
        verbose = param_dict.get('verbose', False)
        
        # We need to re-read image because s1_out only has path
        image_src = Image.open(s1_out['file_name']).convert("RGB")
        w, h = s1_out['width'], s1_out['height']
        
        # Limit proposals — increased from 10 to 50 for LVIS 1203 class coverage
        max_regions = 50
        proposals_to_process = min(len(bg_boxes), max_regions)
        
        for idx in range(proposals_to_process):
            box = bg_boxes[idx]
            x1, y1, x2, y2 = map(int, box.tolist())
            # Scale coords if they were normalized? No, they are absolute in s1_out["background"]["boxes"]?
            # Wait, `extract_rpn_proposals` returns absolute boxes?
            # Let's check logic:
            # extract_rpn_proposals returns boxes from forward hook. M-RCNN usually returns absolute boxes.
            # let's assume absolute.
            
            # Clamp
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if (x2-x1) < 10 or (y2-y1) < 10: continue
            
            crop = image_src.crop((x1, y1, x2, y2))
            
            # Score
            class_idx, confidence, class_name = vlrm.score_region(crop) 
            if confidence >= 0.35:
                s2_bg_boxes.append(box)
                s2_bg_scores.append(confidence)
                s2_bg_labels.append(class_idx)
                if verbose:
                     print(f"[Legacy Stage 2] {class_name} {confidence:.2f}")

    if len(s2_bg_boxes) > 0:
        stage2_out = {
            "boxes": torch.stack(s2_bg_boxes).to(device),
            "scores": torch.tensor(s2_bg_scores, device=device),
            "labels": torch.tensor(s2_bg_labels, device=device)
        }
    else:
        stage2_out = {
            "boxes": torch.empty(0, 4, device=device),
            "scores": torch.empty(0, device=device),
            "labels": torch.empty(0, device=device)
        }
        
    # 3. Run Stage 3
    final_instances = run_stage3_inference(s1_out, stage2_out, param_dict)
    
    return final_instances