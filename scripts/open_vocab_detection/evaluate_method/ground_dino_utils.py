"""
FYP: GroundingDINO utils for COCO Open-Vocabulary Detection
=============================================================

This module provides functions for:
1. Preparing images for GroundingDINO
2. Extracting noun phrases from captions
3. Running inference with proper COCO class mapping (0-indexed)

Key FYP Change: Uses BLIP captions (+ LLM refinement) instead of fixed templates
"""
import numpy as np
import transforms as T2
import torch
import cv2
import os
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import sys

# Add parent scripts directory to path for SRM module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

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
from utils import build_captions_and_token_span, create_positive_map_from_span

# COCO 80 class names (in order of contiguous 0-indexed IDs)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Extended synonyms for better matching
SYNONYMS = {
    'person': ['man', 'woman', 'people', 'child', 'kid', 'boy', 'girl', 'human', 'lady', 'guy', 'player', 'skier', 'surfer', 'rider'],
    'car': ['vehicle', 'automobile', 'auto', 'sedan', 'suv'],
    'truck': ['lorry', 'pickup'],
    'couch': ['sofa', 'settee'],
    'tv': ['television', 'monitor', 'screen', 'display'],
    'cell phone': ['phone', 'cellphone', 'mobile', 'smartphone', 'iphone'],
    'laptop': ['computer', 'notebook', 'macbook'],
    'potted plant': ['plant', 'flower', 'houseplant'],
    'dining table': ['table', 'desk'],
    'traffic light': ['stoplight', 'signal'],
    'sports ball': ['ball', 'soccer ball', 'football', 'basketball', 'baseball'],
    'teddy bear': ['teddy', 'stuffed animal', 'plush'],
    'hot dog': ['hotdog'],
    'fire hydrant': ['hydrant'],
    'stop sign': ['sign'],
    'wine glass': ['glass', 'wineglass'],
    'baseball bat': ['bat'],
    'baseball glove': ['glove', 'mitt'],
    'tennis racket': ['racket', 'racquet'],
    'hair drier': ['hairdryer', 'dryer'],
    'remote': ['remote control'],
}


def prepare_image_for_GDINO(input, device="cuda"):
    """Prepare image for GroundingDINO inference."""
    transform = T2.Compose([
        T2.RandomResize([800], max_size=1333),
        T2.ToTensor(),
        T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_src = Image.open(input["file_name"]).convert("RGB")
    image = np.asarray(image_src)
    image_transformed, _ = transform(image_src, None)
    image_transformed = image_transformed.to(device)
    return image_transformed[None], image


def get_noun_phrases(caption):
    """Extract noun phrases from caption using NLTK."""
    words = word_tokenize(caption)
    pos_tags = nltk.pos_tag(words)
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)
    noun_phrases = []
    for subtree in result.subtrees(filter=lambda t: t.label() == 'NP'):
        noun_phrases.append(' '.join(word for word, tag in subtree.leaves()))
    return noun_phrases


def normalize_caption_to_object(text_refiner, blip_caption):
    """
    Use LLM to normalize BLIP scene captions to object-centric names.
    
    Example:
        Input: "a brown teddy bear sitting on a white couch"
        Output: "brown teddy bear"
    
    Args:
        text_refiner: LLM text refiner
        blip_caption: Raw BLIP caption (may include scene context)
    
    Returns:
        Normalized object name (noun/noun phrase only)
    """
    try:
        # Use text_refiner's refine method with object-extraction prompt
        prompt = f'''Extract ONLY the main object name from this caption. Remove all scene context, locations, and actions.

Caption: "{blip_caption}"

Object name (1-3 words):'''
        
        # Check if text_refiner has a simple text refinement method
        if hasattr(text_refiner, 'refine_text'):
            result = text_refiner.refine_text(prompt)
        elif hasattr(text_refiner, 'generate'):
            result = text_refiner.generate(prompt)
        else:
            # Fallback to heuristic
            return extract_first_noun_phrase(blip_caption)
        
        # Clean up result
        normalized = result.strip().strip('"').strip("'").lower()
        return normalized if normalized else blip_caption
        
    except Exception as e:
        print(f"[Warning] LLM normalization error: {e}")
        return extract_first_noun_phrase(blip_caption)


def extract_first_noun_phrase(caption):
    """
    Heuristic fallback: Extract first substantial noun phrase from caption.
    
    Example:
        "a brown teddy bear on a couch" → "brown teddy bear"
    """
    # Get all noun phrases
    noun_phrases = get_noun_phrases(caption)
    
    if noun_phrases:
        # Return longest noun phrase (likely the main object)
        return max(noun_phrases, key=len)
    
    # Fallback: return first few non-article words
    words = caption.lower().split()
    # Skip articles
    filtered = [w for w in words if w not in ['a', 'an', 'the', 'on', 'in', 'at', 'with']]
    if filtered:
        # Take first 1-3 words as object name
        return ' '.join(filtered[:3])
    
    return "object"



def noun_phrase_to_contiguous_id(noun_phrase):
    """
    Map noun phrase to COCO contiguous class ID (0-indexed, 0-79).
    Returns class ID or -1 if no match.
    
    IMPORTANT: Returns 0-indexed contiguous IDs, NOT COCO category IDs!
    """
    np_lower = noun_phrase.lower().strip()
    
    # Direct matching against class names
    for idx, class_name in enumerate(COCO_CLASSES):
        if class_name in np_lower or np_lower in class_name:
            return idx  # Return 0-indexed contiguous ID
    
    # Synonym matching
    for class_name, syns in SYNONYMS.items():
        for syn in syns:
            if syn in np_lower:
                return COCO_CLASSES.index(class_name)  # Return 0-indexed
    
    return -1  # No match


@torch.no_grad()
def inference_gdino(model, inputs, caption, param_dict):
    """
    Run GroundingDINO inference with caption-based prompts.
    Returns instances with 0-indexed contiguous class IDs (0-79).
    """
    visualize = param_dict["visualize"]
    out_dir = param_dict["out_dir"]
    data_split = param_dict["data_split"]
    device = param_dict["device"]
    ovd_id_to_coco_id = param_dict["ovd_id_to_coco_id"]

    # SAM
    sam = param_dict["sam"]
    resize_transform = param_dict["resize_transform"]

    image, image_src = prepare_image_for_GDINO(inputs[0], device)
    image = image.repeat(1, 1, 1, 1)

    noun_phrases = get_noun_phrases(caption)
    
    if not noun_phrases:
        h, w = inputs[0]['height'], inputs[0]['width']
        result = Instances((h, w))
        result.pred_boxes = Boxes(torch.empty(0, 4))
        result.scores = torch.empty(0)
        result.pred_classes = torch.empty(0, dtype=torch.int64)
        return [{"instances": result}]

    text_prompt, cat2tokenspan = build_captions_and_token_span(noun_phrases, True)
    
    # Use only valid noun phrases
    valid_noun_phrases = [np for np in noun_phrases if np in cat2tokenspan]
    
    if not valid_noun_phrases:
        h, w = inputs[0]['height'], inputs[0]['width']
        result = Instances((h, w))
        result.pred_boxes = Boxes(torch.empty(0, 4))
        result.scores = torch.empty(0)
        result.pred_classes = torch.empty(0, dtype=torch.int64)
        return [{"instances": result}]
    
    tokenizer = model.tokenizer
    tokenspanlist = [cat2tokenspan[cat] for cat in valid_noun_phrases]
    positive_map = create_positive_map_from_span(tokenizer(text_prompt), tokenspanlist)

    with torch.no_grad():
        output = model(image, captions=[text_prompt])

    out_logits = output["pred_logits"]
    out_bbox = output["pred_boxes"]
    prob_to_token = out_logits.sigmoid()

    out_bbox = out_bbox.squeeze(0)  # GPU: stay on device
    curr_prob_to_label = prob_to_token[0] @ positive_map.to(prob_to_token.device).T
    prob_to_label = curr_prob_to_label  # GPU: stay on device

    # GPU Optimization: Increased from 300 to 900 for more candidates, then filter by quality
    topk_values, topk_idxs = torch.topk(prob_to_label.view(-1), 900, 0)
    scores = topk_values
    topk_boxes = topk_idxs // prob_to_label.shape[1]
    labels_np_idx = topk_idxs % prob_to_label.shape[1]

    boxes = out_bbox[topk_boxes]

    h, w = inputs[0]['height'], inputs[0]['width']
    scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
    boxes = boxes * scale
    boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
    
    # ===========================================================================
    # GPU Optimization: Relaxed score threshold for better recall
    # ===========================================================================
    SCORE_THRESHOLD = 0.10  # GPU Optimization: Lowered from 0.15 for better recall
    score_mask = scores >= SCORE_THRESHOLD
    scores = scores[score_mask]
    boxes = boxes[score_mask]
    labels_np_idx = labels_np_idx[score_mask]
    
    if len(scores) == 0:
        result = Instances((h, w))
        result.pred_boxes = Boxes(torch.empty(0, 4))
        result.scores = torch.empty(0)
        result.pred_classes = torch.empty(0, dtype=torch.int64)
        return [{"instances": result}]
    
    # Map noun phrases to 0-indexed contiguous class IDs (0-79)
    contiguous_labels = []
    valid_mask = []
    for idx in range(len(labels_np_idx)):
        np_idx = labels_np_idx[idx].item()
        if np_idx < len(valid_noun_phrases):
            np_text = valid_noun_phrases[np_idx]
            class_id = noun_phrase_to_contiguous_id(np_text)
            if 0 <= class_id < 80:  # Valid COCO class
                contiguous_labels.append(class_id)
                valid_mask.append(True)
            else:
                # Default to class 0 (person) for unmatched
                contiguous_labels.append(0)
                valid_mask.append(True)
        else:
            valid_mask.append(False)
    
    valid_mask = torch.tensor(valid_mask)
    if valid_mask.sum() == 0:
        h, w = inputs[0]['height'], inputs[0]['width']
        result = Instances((h, w))
        result.pred_boxes = Boxes(torch.empty(0, 4))
        result.scores = torch.empty(0)
        result.pred_classes = torch.empty(0, dtype=torch.int64)
        return [{"instances": result}]
    
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    labels = torch.tensor(contiguous_labels, dtype=torch.int64)[:valid_mask.sum()]

    # Ensure all labels are in valid range [0, 79]
    labels = torch.clamp(labels, min=0, max=79)

    # ===========================================================================
    # CRITICAL FIX: Apply NMS per-class to reduce duplicate detections
    # ===========================================================================
    if len(boxes) > 0:
        # GPU: ensure all on same device for batched_nms
        boxes = boxes.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        keep = batched_nms(boxes, scores, labels, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
    
    # NOTE: Removed MinMaxScaler - it destroys confidence ranking across images

    # SAM refinement (disabled for 5-hour target)
    if sam is not None and len(boxes) > 0:
        boxes = boxes.to(sam.device)
        curr_image = cv2.imread(inputs[0]['file_name'])
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)

        img_shape = curr_image.shape[:2]
        curr_image = resize_transform.apply_image(curr_image)
        curr_image = torch.as_tensor(curr_image, device=sam.device).permute(2, 0, 1).contiguous()

        sam_box_prompts = resize_transform.apply_boxes_torch(boxes, img_shape)

        # GPU Optimization: Process SAM in larger batches for better GPU utilization
        sam_batch_size = 50  # Increased from 25 to 50
        sam_refined_boxes_list = []
        sam_scores_list = []

        for i in range(0, len(sam_box_prompts), sam_batch_size):
            batch_sam_box_prompts = sam_box_prompts[i:i + sam_batch_size]

            if len(batch_sam_box_prompts) == 0:
                continue

            batched_input = [{
                "image": curr_image,
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
            sam_scores = torch.cat(sam_scores_list, dim=0)
            sam_scores = sam_scores.squeeze(1)  # GPU: stay on device for SRM

            # ==================================================================
            # Score Refinement Module (SRM) - Paper Algorithm 1
            # ==================================================================
            if len(sam_scores) > 0:
                print(f"[SRM] Refining {len(scores)} scores using SAM mask quality...")
                
                # Apply SRM refinement (Algorithm 1: MinMaxScaler + multiplication)
                srm = ScoreRefinementModule(per_image_norm=True)
                scores = srm.refine_scores(scores, sam_scores)
                
                print(f"[SRM] ✓ Scores refined: range=[{scores.min():.3f}, {scores.max():.3f}]")
            
            # Keep top 100
            if len(scores) > 100:
                topk_scores, topk_idxs = torch.topk(scores, 100)
                boxes = sam_refined_boxes[topk_idxs]
                labels = labels[topk_idxs]
                scores = topk_scores
            else:
                boxes = sam_refined_boxes
    
    # Final validation: ensure all class IDs are in [0, 79]
    labels = torch.clamp(labels, min=0, max=79)
    
    # Build result
    result = Instances((h, w))
    result.pred_boxes = Boxes(boxes.cpu() if torch.is_tensor(boxes) else boxes)
    result.scores = scores.cpu() if torch.is_tensor(scores) else scores
    result.pred_classes = labels.cpu() if torch.is_tensor(labels) else labels

    return [{"instances": result}]


@torch.no_grad()
def inference_maskrcnn(maskrcnn_model, inputs, device):
    """
    Run TorchVision Mask-RCNN inference to get RPN proposals.
    
    CRITICAL: This function extracts RPN proposals (raw object candidates),
    NOT the final closed-set detections. This enables novel object discovery.
    
    TorchVision Mask-RCNN format:
    - Expects RGB images as float tensors [0, 1] in CHW format
    - RPN proposals are extracted via forward hooks or model internals
    
    Returns:
        boxes: Proposal bounding boxes (500-1000 per image)
        scores: Objectness scores (0-1)
        classes: Placeholder (RPN is class-agnostic, set to 0)
    """
    if maskrcnn_model is None:
        print("[Mask-RCNN] Model is None!")
        eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
        return eb, es, ec, eb, es, ec
    
    try:
        height = inputs[0]["height"]
        width = inputs[0]["width"]
        file_name = inputs[0].get("file_name")
        
        # =======================================================================
        # TorchVision expects RGB float tensor [0, 1] in CHW format
        # =======================================================================
        import cv2
        
        if file_name and os.path.exists(file_name):
            # Load image from file as RGB
            bgr_image = cv2.imread(file_name)
            if bgr_image is None:
                print(f"[Mask-RCNN] Failed to load image: {file_name}")
                eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
                return eb, es, ec, eb, es, ec
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # Convert to float tensor [0, 1], HWC to CHW
            image = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
            image = image.to(device)
            
            print(f"[Mask-RCNN] Loaded image: {image.shape}, range=[{image.min():.2f}, {image.max():.2f}]")
        else:
            # Fallback: use tensor from dataloader
            image = inputs[0]["image"].to(device)
            
            # Ensure correct format
            if image.dtype == torch.uint8:
                image = image.float() / 255.0
            
            # Detectron2 uses BGR, TorchVision uses RGB - flip channels
            if image.shape[0] == 3:
                image = image.flip(0)  # BGR to RGB
            
            print(f"[Mask-RCNN] Using dataloader image: {image.shape}")
        
        print(f"[Mask-RCNN] Running inference on image {width}x{height}...")
        
        # =======================================================================
        # SOLUTION 1: Hook into RPN to extract proposals before filtering
        # =======================================================================
        proposals_captured = []
        scores_captured = []
        
        def capture_proposals(module, input, output):
            """Hook to capture RPN proposals before they go to ROI head."""
            if isinstance(output, tuple) and len(output) >= 2:
                boxes_list, scores_list = output[0], output[1]
                if boxes_list and len(boxes_list) > 0:
                    proposals_captured.extend([b for b in boxes_list])  # GPU: keep on model device
                    if scores_list and len(scores_list) > 0:
                        scores_captured.extend([s for s in scores_list])
        
        # Register hook on RPN
        hook = None
        if hasattr(maskrcnn_model, 'rpn'):
            hook = maskrcnn_model.rpn.register_forward_hook(capture_proposals)
        
        # Run full inference to trigger RPN
        with torch.no_grad():
            # TorchVision expects list of tensors
            outputs = maskrcnn_model([image])
        
        # Remove hook
        if hook:
            hook.remove()
        
        # Check if we captured RPN proposals
        if proposals_captured:
            all_boxes = torch.cat(proposals_captured, dim=0)
            if scores_captured:
                all_scores = torch.cat(scores_captured, dim=0).flatten()
                # Sigmoid if they're logits
                if all_scores.min() < 0:
                    all_scores = all_scores.sigmoid()
            else:
                all_scores = torch.ones(len(all_boxes))
            
            classes = torch.zeros(len(all_boxes), dtype=torch.int64)
            
            # --- Extract KNOWN objects from final outputs ---
            known_boxes, known_scores, known_classes = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
            if outputs and len(outputs) > 0:
                output = outputs[0]
                if 'boxes' in output and len(output['boxes']) > 0:
                    known_boxes = output['boxes'].cpu()
                    known_scores = output['scores'].cpu()
                    known_classes = output['labels'].cpu()
                    mask = known_scores >= 0.5  # Only confident knowns
                    known_boxes = known_boxes[mask]
                    known_scores = known_scores[mask]
                    known_classes = known_classes[mask]
            
            print(f"[Mask-RCNN] ✓ RPN captured {len(all_boxes)} proposals via hook")
            if len(known_boxes) > 0:
                print(f"[Mask-RCNN] ✓ Got {len(known_boxes)} known objects (>0.5)")
            return all_boxes, all_scores, classes, known_boxes, known_scores, known_classes
        
        # =======================================================================
        # SOLUTION 2: Use final detections with very LOW threshold
        # TorchVision returns list of dicts with 'boxes', 'labels', 'scores'
        # =======================================================================
        if outputs and len(outputs) > 0:
            output = outputs[0]
            
            if 'boxes' in output and len(output['boxes']) > 0:
                boxes = output['boxes'].cpu()
                scores = output['scores'].cpu()
                classes = output['labels'].cpu()
                
                # Accept all detections, even low confidence
                mask = scores >= 0.0  # Keep everything
                boxes = boxes[mask]
                scores = scores[mask]
                classes = classes[mask]
                
                if len(boxes) > 0:
                    print(f"[Mask-RCNN] ✓ Got {len(boxes)} final detections")
                    print(f"[Mask-RCNN] Score range: [{scores.min():.3f}, {scores.max():.3f}]")
                    eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
                    return boxes, scores, classes, boxes, scores, classes
        
        # =======================================================================
        # SOLUTION 3: Access RPN internal state directly
        # =======================================================================
        if hasattr(maskrcnn_model, 'rpn') and hasattr(maskrcnn_model.rpn, 'head'):
            try:
                # Get backbone features
                images = [image]
                features = maskrcnn_model.backbone(maskrcnn_model.transform(images)[0].tensors)
                
                # Get feature list
                if isinstance(features, dict):
                    features_list = list(features.values())
                else:
                    features_list = [features]
                
                # Get image shapes
                image_shapes = [(image.shape[1], image.shape[2])]
                
                # Run RPN head
                objectness, pred_bbox_deltas = maskrcnn_model.rpn.head(features_list)
                
                # Generate anchors
                anchors = maskrcnn_model.rpn.anchor_generator(images, features_list)
                
                # Decode proposals
                from torchvision.models.detection.rpn import concat_box_prediction_layers
                objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
                
                # Apply sigmoid to objectness
                objectness_probs = objectness.sigmoid()
                
                # Get top proposals
                num_anchors = objectness_probs.shape[0]
                keep = torch.argsort(objectness_probs.flatten(), descending=True)[:2000]
                
                boxes = anchors[0][keep]  # Use anchors as proposals
                scores = objectness_probs.flatten()[keep]
                classes = torch.zeros(len(boxes), dtype=torch.int64)
                
                print(f"[Mask-RCNN] ✓ RPN direct extraction: {len(boxes)} proposals")
                eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
                return boxes.cpu(), scores.cpu(), classes, eb, es, ec
                
            except Exception as e:
                print(f"[Mask-RCNN] Direct RPN extraction failed: {e}")
                import traceback
                traceback.print_exc()
        
        print("[Mask-RCNN] ❌ No proposals generated")
        eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
        return eb, es, ec, eb, es, ec
        
    except Exception as e:
        print(f"[Mask-RCNN] Inference error: {e}")
        import traceback
        traceback.print_exc()
        eb, es, ec = torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
        return eb, es, ec, eb, es, ec




def extract_background_proposals(gdino_boxes, maskrcnn_boxes, maskrcnn_scores, maskrcnn_classes, iou_threshold=0.3):
    """
    Extract background proposals from Mask-RCNN that don't overlap with GDINO detections.
    
    Args:
        gdino_boxes: Boxes detected by GroundingDINO (foreground)
        maskrcnn_boxes: All boxes from Mask-RCNN
        maskrcnn_scores: Confidence scores from Mask-RCNN
        maskrcnn_classes: Class predictions from Mask-RCNN
        iou_threshold: Maximum IoU to consider as background (default 0.3)
    
    Returns:
        background_boxes, background_scores, background_classes
    """
    if maskrcnn_boxes is None or len(maskrcnn_boxes) == 0:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    
    if gdino_boxes is None or len(gdino_boxes) == 0:
        # All Mask-RCNN proposals are background
        return maskrcnn_boxes, maskrcnn_scores, maskrcnn_classes
    
    # Compute IoU between Mask-RCNN and GDINO boxes
    gdino_boxes_tensor = gdino_boxes.tensor if hasattr(gdino_boxes, 'tensor') else gdino_boxes
    
    ious = pairwise_iou(Boxes(maskrcnn_boxes), Boxes(gdino_boxes_tensor))
    
    # Get max IoU for each Mask-RCNN box
    max_ious, _ = ious.max(dim=1)
    
    # Background = low IoU with all GDINO boxes
    background_mask = max_ious < iou_threshold
    
    bg_boxes = maskrcnn_boxes[background_mask]
    bg_scores = maskrcnn_scores[background_mask]
    bg_classes = maskrcnn_classes[background_mask]
    
    return bg_boxes, bg_scores, bg_classes


def label_background_regions_with_clip(
    image_path, 
    background_boxes,
    captioner,  # BLIP for cropped proposals
    text_refiner,  # LLM for caption normalization
    clip_model,
    clip_preprocess,
    coco_text_features,
    tokenizer_clip,
    device,
    glip_model=None,  # NEW: GLIP for grounding filter
    max_regions=50
):
    """
    Label background regions using BLIP captioning + GLIP grounding filter + CLIP semantic matching.
    
    Pipeline:
    1. BLIP generates caption for cropped region
    2. Ollama LLM normalizes caption to object name
    3. GLIP validates region quality (grounding filter) ← NEW!
    4. CLIP performs semantic matching on validated regions only
    
    Returns:
        labeled_boxes: Tensor (N, 4)
        labeled_scores: Tensor (N,) - CLIP similarity scores
        labeled_classes: Tensor (N,) - COCO class IDs (0-79)
    """
    import torch.nn.functional as F
    import numpy as np
    
    # Caption blacklist for generic/scene descriptions
    CAPTION_BLACKLIST = {"background", "image", "picture", "photo", "scene", "blur", "blurry"}
    
    if captioner is None or clip_model is None or len(background_boxes) == 0:
        if captioner is None:
            print("[Stage 2] WARNING: No captioner available - BLIP required for Stage-2!")
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    
    # Load image
    try:
        pil_image = Image.open(image_path).convert("RGB")
        cropped_img_arr = np.array(pil_image) # For GLIP
    except Exception as e:
        print(f"[Stage 2] Error loading image: {e}")
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    
    # Limit regions for efficiency
    num_regions = min(len(background_boxes), max_regions)
    
    valid_boxes = []
    scores_list = []
    classes_list = []
    
    for i in range(num_regions):
        box = background_boxes[i]
        x1, y1, x2, y2 = map(int, box.tolist())
        
        # Ensure valid crop coordinates
        x1, x2 = max(0, x1), min(pil_image.width, x2)
        y1, y2 = max(0, y1), min(pil_image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        try:
            # Crop region
            crop = pil_image.crop((x1, y1, x2, y2))
            
            # =====================================================
            # Step 1: BLIP Caption Generation (Object-Level, NOT Scene)
            # Cropped image → BLIP → Object caption
            # =====================================================
            object_caption = "object"  # Default fallback
            
            try:
                # BLIP generates caption from CROPPED region (not full image!)
                blip_result = captioner.inference(crop)  # PIL Image
                raw_caption = blip_result.get('caption', 'object')
                print(f"[BLIP Stage-2] Proposal {i} raw: '{raw_caption}'")
                
                #LLM Normalization - Extract object name only
                if text_refiner is not None:
                    try:
                        # TextRefiner uses .inference(query, controls) method
                        result = text_refiner.inference(raw_caption, {})
                        normalized = result.get('caption', raw_caption)
                        if normalized and normalized != raw_caption:
                            print(f"[LLM] '{raw_caption}' → '{normalized}'")
                            object_caption = normalized
                        else:
                            object_caption = raw_caption
                    except Exception as e:
                        object_caption = raw_caption
                else:
                    object_caption = raw_caption
                
                # Quick blacklist check (word-level)
                caption_words = object_caption.lower().split()
                if any(word in CAPTION_BLACKLIST for word in caption_words):
                    continue
                
            except Exception as e:
                print(f"[Warning] BLIP failed for proposal {i}: {e}")
                object_caption = "object"
            
            # ==================================================================
            # Step 2: GLIP Grounding Filter (validates region quality)
            # ==================================================================
            if glip_model is not None:
                try:
                    # Crop region for GLIP
                    cropped_region = cropped_img_arr[int(y1):int(y2), int(x1):int(x2)]
                    
                    # GLIP validates if this is a real object (not noise/background)
                    glip_result = glip_model.inference(cropped_region, object_caption)
                    
                    if len(glip_result) > 0:
                        glip_scores = glip_result.get_field("scores")
                        if len(glip_scores) > 0:
                            glip_score = glip_scores[0].item()
                            
                            # GLIP grounding threshold: 0.3
                            if glip_score < 0.3:
                                print(f"[GLIP] ❌ Low {glip_score:.2f}")
                                continue  # Reject
                            else:
                                print(f"[GLIP] ✓ {glip_score:.2f}")
                except Exception as e:
                    pass  # Continue without GLIP on error
            
            # ==================================================================
            # Step 3: CLIP Semantic Matching (on GLIP-validated regions only)
            # ==================================================================
            # CLIP Text Encoding: VLM caption → Text embedding
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_tokens = tokenizer_clip([object_caption]).to(device)
                caption_text_embedding = clip_model.encode_text(text_tokens)
                caption_text_embedding = F.normalize(caption_text_embedding, dim=-1)
            
            # =====================================================
            # CLIP Image Encoding: Cropped region → Image embedding
            # =====================================================
            with torch.no_grad(), torch.cuda.amp.autocast():
                crop_tensor = clip_preprocess(crop).unsqueeze(0).to(device)
                crop_image_embedding = clip_model.encode_image(crop_tensor)
                crop_image_embedding = F.normalize(crop_image_embedding, dim=-1)
            
            # =====================================================
            # Similarity Computation: VLM caption + Image → COCO classes
            # Hybrid approach: Cosine + Euclidean for robustness
            # =====================================================
            # Cosine similarity: VLM caption text ↔ COCO class embeddings
            text_to_class_cosine = (caption_text_embedding @ coco_text_features.T).squeeze(0)
            
            # Cosine similarity between crop image and COCO classes
            image_to_class_cosine = (crop_image_embedding @ coco_text_features.T).squeeze(0)
            
            # Euclidean distance (lower is better)
            image_to_class_euclidean = torch.cdist(
                crop_image_embedding, 
                coco_text_features, 
                p=2
            ).squeeze(0)
            
            # Convert Euclidean to similarity score (higher is better)
            image_to_class_euclidean_sim = 1.0 / (1.0 + image_to_class_euclidean)
            
            # Hybrid: Combine cosine and Euclidean (weighted average)
            image_sim_hybrid = (image_to_class_cosine * 0.5 + image_to_class_euclidean_sim * 0.5)
            text_sim = text_to_class_cosine
            
            # Final combined similarity
            combined_sim = (text_sim + image_sim_hybrid) / 2
            
            # Get best matching class
            best_score, best_class = combined_sim.max(dim=0)
            
            # Log CLIP similarity score
            class_name = COCO_CLASSES[best_class.item()] if best_class.item() < len(COCO_CLASSES) else "unknown"
            print(f"[Stage 2] Cropping proposal {i}")
            print(f'[BLIP] Caption: "{object_caption}"')
            print(f"[CLIP] Similarity score: {best_score.item():.2f} -> {class_name}")
            
            # ===========================================================================
            # QUALITY FILTER: Minimum CLIP score threshold
            # Speed optimization: Lowered from 0.55 to 0.50 for better recall
            # ===========================================================================
            MIN_CLIP_SCORE = 0.50  # GPU Optimization: Lower threshold, rely on SRM
            if best_score.item() < MIN_CLIP_SCORE:
                continue
            
            # ===========================================================================
            # QUALITY FILTER 2: Caption blacklist
            # Reject generic/invalid captions that describe scene, not objects
            # ===========================================================================
            CAPTION_BLACKLIST = [
                "background", "image", "picture", "photo", "scene", 
                "blurry", "blur", "close up", "closeup", "view",
                "area", "surface", "part", "piece", "section"
            ]
            caption_lower = object_caption.lower()
            is_blacklisted = any(word in caption_lower for word in CAPTION_BLACKLIST)
            if is_blacklisted:
                print(f"[Stage 2] ❌ Rejected: Caption '{object_caption}' contains blacklisted word")
                continue
            
            valid_boxes.append(box)
            scores_list.append(best_score.cpu().item())
            classes_list.append(best_class.cpu().item())
            
        except Exception as e:
            print(f"[Stage 2] Error processing region {i}: {e}")
            continue
    
    if not valid_boxes:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0, dtype=torch.int64)
    
    bg_boxes = torch.stack(valid_boxes)
    bg_scores = torch.tensor(scores_list, dtype=torch.float32)
    bg_classes = torch.tensor(classes_list, dtype=torch.int64)
    
    return bg_boxes, bg_scores, bg_classes


# Legacy function for backwards compatibility
def caption_background_regions(image_path, background_boxes, captioner, text_refiner, device, max_regions=10):
    """Legacy function - use label_background_regions_with_clip instead."""
    return []


def get_class_from_caption(caption):
    """
    Extract the best matching COCO class from a caption using noun phrase matching.
    Returns (class_id, class_name) or (-1, None) if no match.
    """
    noun_phrases = get_noun_phrases(caption)
    
    for np_text in noun_phrases:
        class_id = noun_phrase_to_contiguous_id(np_text)
        if class_id >= 0:
            return class_id, COCO_CLASSES[class_id]
    
    words = caption.lower().split()
    for word in words:
        for idx, class_name in enumerate(COCO_CLASSES):
            if word in class_name or class_name in word:
                return idx, class_name
    
    return -1, None
