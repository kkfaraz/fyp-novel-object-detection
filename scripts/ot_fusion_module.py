"""
Hybrid Matching (OT + Hungarian) Fusion Module
==============================================

Replaces general OT (Sinkhorn) with the method from:
"Improved object detection method for unmanned driving based on Transformers"

Logic:
1. Construct Cost Matrix C = alpha * L_cls + beta * L_reg
   - L_cls: Focal Loss
   - L_reg: GIOU + L1
2. Hybrid Matching:
   - Branch 1: Optimal Transport (Dynamic Top-k)
   - Branch 2: Hungarian Matching
3. Fusion:
   - Use the matching results to align and boost scores.
   - Per paper, OT matching "expands positive samples" for dense occlusion.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou

def compute_focal_loss_cost(
    cls_pred: torch.Tensor,
    cls_target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute Focal Loss Cost between predictions and targets.
    Ref: Equation (17) Lcls
    
    Args:
        cls_pred: (N, num_classes) predicted class probabilities (sigmoid applied)
        cls_target: (M,) target class labels
        
    Returns:
        cost: (N, M)
    """
    N, C = cls_pred.shape
    M = cls_target.shape[0]
    
    # Expand pred to (N, M, C)
    # This is expensive. Instead, gather relevant scores.
    # Cost = -alpha * (1-p)^gamma * log(p)   if y=1
    #      = -(1-alpha) * p^gamma * log(1-p) if y=0
    
    # We want cost matrix (N, M).
    # Each cell (i, j) is loss between pred_i and target_j.
    # target_j has class c_j.
    # So for pred_i, we look at prob of class c_j.
    
    # Gather probs for target classes: (N, M)
    # cls_pred: (N, C)
    # cls_target: (M,) -> indices
    target_probs = cls_pred[:, cls_target] # (N, M)
    
    # Focal Loss terms
    # Positive term (for the target class)
    # cost_pos = -alpha * (1 - p_t)^gamma * log(p_t)
    cost_pos = -alpha * ((1 - target_probs) ** gamma) * torch.log(target_probs + 1e-8)
    
    # Negative term (for all OTHER classes)
    # This is usually constant or handled differently in DETR cost?
    # DETR Focal Loss Cost is usually just -(alpha * (1-p)^gamma * log(p)) for the target class
    # minus -(1-alpha) * p^gamma * log(1-p) for the target class?
    # Paper Eq (17) sums over classes.
    # For cost matrix matching, we typically only consider the target class probability 
    # and neglect the background suppression for non-target classes (as they are common).
    # Standard DETR implementation uses:
    # cost = -alpha * (1 - p_i)^gamma * log(p_i)
    
    return cost_pos

def compute_l1_cost(boxes_pred: torch.Tensor, boxes_target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 cost.
    Args:
        boxes_pred: (N, 4) xyxy
        boxes_target: (M, 4) xyxy
    """
    # Normalize? Assuming inputs are consistent.
    # L1 = |b1 - b2|
    # (N, M, 4)
    # Using cdist with p=1
    return torch.cdist(boxes_pred, boxes_target, p=1)

def compute_giou_cost(boxes_pred: torch.Tensor, boxes_target: torch.Tensor) -> torch.Tensor:
    """
    Compute GIOU cost. 1 - GIOU.
    Args:
        boxes_pred: (N, 4) xyxy
        boxes_target: (M, 4) xyxy
    """
    # generalized_box_iou returns (N, M)
    giou = generalized_box_iou(boxes_pred, boxes_target)
    return 1 - giou

def dynamic_topk_matching(
    cost_matrix: torch.Tensor,
    iou_matrix: torch.Tensor,
    k_min: int = 1
) -> torch.Tensor:
    """
    Dynamic Top-k (OTA / SimOTA) matching strategy.
    
    Args:
        cost_matrix: (N, M) Cost between N preds and M targets
        iou_matrix: (N, M) IoU between N preds and M targets
        
    Returns:
        matching_mask: (N, M) 1 if matched, 0 otherwise
    """
    N, M = cost_matrix.shape
    matching_matrix = torch.zeros_like(cost_matrix)
    
    if N == 0 or M == 0:
        return matching_matrix
    
    # For each ground truth (col j), select dynamic k
    # k_j = sum of IoUs for this GT (clamped)
    
    # Top-k IoUs
    k_top = min(10, N)
    topk_ious, _ = torch.topk(iou_matrix, k_top, dim=0) # (k_top, M)
    dynamic_ks = torch.clamp(topk_ious.sum(dim=0).int(), min=k_min) # (M,)
    
    for j in range(M):
        k = min(dynamic_ks[j].item(), N)
        if k == 0:
            continue
        # Select top k candidates with smallest cost
        _, indices = torch.topk(cost_matrix[:, j], k, largest=False)
        matching_matrix[indices, j] = 1.0
        
    # Enforce one-to-one constraint for PREDICTIONS? 
    # OTA allows multiple preds per GT. But one pred typically matches only one GT.
    # If a pred matches multiple GTs, assign to closest.
    
    # Check if any row (pred) has > 1 match
    multi_matches = matching_matrix.sum(dim=1) > 1
    if multi_matches.any():
        # Resolve ambiguity: take min cost
        for i in torch.where(multi_matches)[0]:
            # Get matched cols
            cols = torch.where(matching_matrix[i] == 1)[0]
            # Find min cost
            min_cost_idx = cols[torch.argmin(cost_matrix[i, cols])]
            matching_matrix[i] = 0
            matching_matrix[i, min_cost_idx] = 1.0
            
    return matching_matrix

def hybrid_match_and_fuse(
    rcnn_boxes: torch.Tensor,
    rcnn_scores: torch.Tensor,
    rcnn_classes: torch.Tensor,
    known_boxes: torch.Tensor = None,
    known_scores: torch.Tensor = None,
    known_classes: torch.Tensor = None,
    gdino_boxes: torch.Tensor = None,
    gdino_scores: torch.Tensor = None,
    gdino_classes: torch.Tensor = None,
    iou_weight: float = 0.6,
    semantic_weight: float = 0.4,
    sinkhorn_reg: float = 0.1, # Legacy param, unused
    sinkhorn_iters: int = 50, # Legacy param, unused
    boost_factor: float = 1.5,
    penalty_factor: float = 1.0,
    param_dict: Optional[Dict[str, Any]] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Hybrid Matching Fusion.
    
    Treats GDINO detections as "Targets" and RCNN/VLRM as "Predictions" 
    (or vice versa - here RCNN->Preds, GDINO->Targets seems logical for "Alignment").
    
    Actually, GDINO -> high recall/precision? RCNN -> high quantity?
    Fusion usually aligns the weaker to the stronger.
    Let's align RCNN (Pred) to GDINO (Target).
    """
    
    verbose = param_dict.get('verbose', False) if param_dict else False
    
    # =========================================================================
    # GPU: Determine canonical device from non-empty input tensors
    # =========================================================================
    device = rcnn_boxes.device
    
    # Ensure all inputs are on the same device
    rcnn_scores = rcnn_scores.to(device)
    rcnn_classes = rcnn_classes.to(device)
    
    # Handle Defaults — all on the canonical device
    if known_boxes is None: known_boxes = torch.empty(0, 4, device=device)
    else: known_boxes = known_boxes.to(device)
    if known_scores is None: known_scores = torch.empty(0, device=device)
    else: known_scores = known_scores.to(device)
    if known_classes is None: known_classes = torch.empty(0, dtype=torch.int64, device=device)
    else: known_classes = known_classes.to(device)
    
    if gdino_boxes is None: gdino_boxes = torch.empty(0, 4, device=device)
    else: gdino_boxes = gdino_boxes.to(device)
    if gdino_scores is None: gdino_scores = torch.empty(0, device=device)
    else: gdino_scores = gdino_scores.to(device)
    if gdino_classes is None: gdino_classes = torch.empty(0, dtype=torch.int64, device=device)
    else: gdino_classes = gdino_classes.to(device)

    # Check empty
    N_bg = len(rcnn_boxes)
    N_known = len(known_boxes)
    M = len(gdino_boxes)
    
    if (N_bg + N_known) == 0 or M == 0:
        # Fallback to concat
        return torch.cat([rcnn_boxes, known_boxes, gdino_boxes], dim=0), \
               torch.cat([rcnn_scores, known_scores, gdino_scores], dim=0), \
               torch.cat([rcnn_classes, known_classes, gdino_classes], dim=0)

    device = gdino_boxes.device
    
    # FIX: DO NOT combine known detections with background proposals for OT matching.
    # Known detections come from Mask-RCNN (trained on COCO, 80 classes) while GDINO
    # classes are 1203 LVIS classes. Their class indices are from DIFFERENT semantic
    # spaces — comparing them directly causes the mismatched penalty to suppress ALL
    # known detections, dropping Known AP from ~45 to ~34.
    #
    # Instead:
    #   - Match only VLRM background proposals (rcnn_boxes) against GDINO
    #   - Keep known detections separate with a guaranteed boost
    #   - Concatenate everything at the end
    
    # Use only VLRM background proposals for OT matching against GDINO
    preds_boxes = rcnn_boxes.to(device) if N_bg > 0 else torch.empty(0, 4, device=device)
    preds_scores = rcnn_scores.to(device) if N_bg > 0 else torch.empty(0, device=device)
    preds_classes = rcnn_classes.to(device) if N_bg > 0 else torch.empty(0, dtype=torch.int64, device=device)
    
    N = len(preds_boxes)
    
    # Prepare Inputs for Cost Matrix
    # 1. Classification Cost (Focal)
    # We need class probabilities for RCNN. 
    # RCNN returns score + class_idx. 
    # We construct "fake" prob vector: score at class_idx, 0 elsewhere?
    # Or just use score difference?
    # Paper Eq (16) Cost = alpha * Lcls + beta * Lreg
    
    # Assume RCNN scores approx probabilities for the assigned class.
    # Lcls(p_j, g_i) = Focal Loss.
    # If classes match: Focal(score, 1). If diff: Focal(0, 1) -> Large cost.
    
    # Cost Matrix Construction
    # Semantic Cost (Paper Eq 20/21/22 uses "Class Accuracy Weight")
    # Actually, simpler: compute L_cls between pred i and target j.
    # If classes different, cost is high.
    
    # Let's map classes to indices or verify equality.
    # We'll use a simplified cost: 
    # Cost = beta * (L_giou + L_l1) + alpha * L_cls
    # L_cls = Focal Loss.
    
    # Normalized coords for L1?
    # Boxes are absolute xyxy.
    # L1 cost on absolute might be large. Normalize by image size?
    # We don't have image size here easily.
    # Let's use GIOU primarily (scale invariant).
    # And L1 on raw coords (maybe scaled).
    
    # Costs
    cost_giou = compute_giou_cost(preds_boxes, gdino_boxes)
    # L1 cost normalized by image diagonal (not hardcoded 1000 which breaks for non-standard image sizes)
    img_h = param_dict.get('img_h', 1000) if param_dict else 1000
    img_w = param_dict.get('img_w', 1000) if param_dict else 1000
    img_diag = max(1.0, (img_h**2 + img_w**2)**0.5)
    cost_l1 = compute_l1_cost(preds_boxes, gdino_boxes) / img_diag
    
    # Semantic Cost (Focal Loss)
    # Cost = alpha * FL(p, target).
    # If match: FL(p, 1) = -alpha * (1-p)^gamma * log(p)
    # If mismatch: Penalty (e.g. FL(0, 1) or just a large constant)
    
    # Calculate FL for all pred scores assuming they match
    # FL_match = -0.25 * (1 - s)^2 * log(s)
    # We use stable formulation
    p = preds_scores.clamp(1e-6, 1-1e-6)
    fl_match = -0.25 * ((1 - p) ** 2) * torch.log(p) # (N,)
    
    # For mismatch, we assume we missed the target class (score ~ 0)
    # fl_mismatch = -0.25 * (1 - 0)^2 * log(0) -> infinity
    # We use a large constant cost for class mismatch
    mismatch_penalty = 2.0 
    
    # Create (N, M) matrix
    # If c_i == c_j: cost = fl_match[i]
    # If c_i != c_j: cost = mismatch_penalty
    
    class_match = (preds_classes[:, None] == gdino_classes[None, :]) # (N, M)
    cost_cls = torch.where(class_match, fl_match[:, None], torch.tensor(mismatch_penalty, device=device))
    
    # Total Cost
    # Paper uses alpha, beta weights
    alpha = semantic_weight
    beta = iou_weight
    
    C = alpha * cost_cls + beta * (cost_giou + cost_l1)
    
    # Branch 1: Optimal Transport (Dynamic Top-k)
    iou_matrix = box_iou(preds_boxes, gdino_boxes)
    ot_mask = dynamic_topk_matching(C, iou_matrix, k_min=1)
    
    # Branch 2: Hungarian Matching
    # min sum(C_ij * x_ij)
    # Scipy linear_sum_assignment cant handle GPU tensors
    C_cpu = C.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(C_cpu)
    hungarian_mask = torch.zeros_like(C)
    hungarian_mask[row_ind, col_ind] = 1.0
    
    # Paper uses Weighted Loss: tau = alpha*H + beta*O
    # Hybrid Matching Fusion Logic:
    # 1. OT Mask (Dynamic Top-K) finds multiple positive samples.
    # 2. Hungarian Mask finds the absolute best 1-to-1 strict matches.
    # We combine them to assign confidence levels.
    
    # Get match counts from OT
    ot_preds_match = ot_mask.sum(dim=1)
    ot_gdino_match = ot_mask.sum(dim=0)
    
    # Get match counts from Hungarian
    # (Since it's 1-to-1, sum is either 0 or 1)
    h_preds_match = hungarian_mask.sum(dim=1)
    h_gdino_match = hungarian_mask.sum(dim=0)
    
    # Score Adjustment Strategy:
    # Per LaTeX paper Section III-C (Score Adjustment):
    # Detections that participate in at least one match receive a score boost (multiplied by 1.5),
    # while unmatched detections receive a penalty (multiplied by 0.7).
    #
    # We construct multipliers:
    # If matched by BOTH        -> boost_factor (e.g. 1.5)
    # If matched by OT ONLY     -> midway boost (e.g. 1.2)
    # If matched by NONE        -> penalty_factor (e.g. 0.7)
    
    def calculate_multiplier(ot_counts, h_counts):
        both_match = (ot_counts > 0) & (h_counts > 0)
        ot_only = (ot_counts > 0) & (h_counts == 0)

        # Base multiplier is penalty_factor (e.g., 0.7)
        multipliers = torch.full_like(ot_counts.float(), penalty_factor, dtype=torch.float32)

        # OT only matches get medium boost (1.2 = halfway between 0.7 and 1.5)
        mid_mult = penalty_factor + (boost_factor - penalty_factor) * 0.625
        multipliers[ot_only] = mid_mult

        # Both matches get full boost multiplier (1.5)
        multipliers[both_match] = boost_factor

        return multipliers

    preds_multipliers = calculate_multiplier(ot_preds_match, h_preds_match)
    gdino_multipliers = calculate_multiplier(ot_gdino_match, h_gdino_match)

    # Apply multiplication transformation and clamp to [0.0, 1.0]
    preds_scores_adj = (preds_scores * preds_multipliers).clamp(0.0, 1.0)
    gdino_scores_adj = (gdino_scores * gdino_multipliers).clamp(0.0, 1.0)
    
    if verbose:
        both_p = ((ot_preds_match > 0) & (h_preds_match > 0)).sum()
        ot_only_p = ((ot_preds_match > 0) & (h_preds_match == 0)).sum()
        print(f"[Hybrid Match] Preds ({N}): {both_p} Both, {ot_only_p} OT-Only, {N - both_p - ot_only_p} Penalized")
        
        both_g = ((ot_gdino_match > 0) & (h_gdino_match > 0)).sum()
        ot_only_g = ((ot_gdino_match > 0) & (h_gdino_match == 0)).sum()
        print(f"[Hybrid Match] GDINO ({M}): {both_g} Both, {ot_only_g} OT-Only, {M - both_g - ot_only_g} Penalized")
    
    # Concatenate matched predictions
    fused_boxes = torch.cat([preds_boxes, gdino_boxes], dim=0)
    fused_scores = torch.cat([preds_scores_adj, gdino_scores_adj], dim=0)
    fused_classes = torch.cat([preds_classes, gdino_classes], dim=0)
    
    # Concatenate known detections without score modification.
    # Known detections from Mask-RCNN are high quality; boosting them ×1.5 caused
    # them to dominate NMS and suppress novel Stage-2 detections (Bug #8).
    if N_known > 0:
        known_scores_adj = known_scores.clamp(0.0, 1.0)  # No boost
        fused_boxes = torch.cat([known_boxes.to(device), fused_boxes], dim=0)
        fused_scores = torch.cat([known_scores_adj, fused_scores], dim=0)
        fused_classes = torch.cat([known_classes.to(device), fused_classes], dim=0)
        if verbose:
            print(f"[Hybrid Match] Known ({N_known}): concatenated without score modification")
    
    return fused_boxes, fused_scores, fused_classes

# Smart wrapper for external calls
def ot_fusion(*args, **kwargs):
    """
    Handles both legacy (6-arg) and modern (9-arg) calls.
    """
    if len(args) == 6:
        # Legacy: (rcnn_b, rcnn_s, rcnn_c, gdino_b, gdino_s, gdino_c)
        # We map gdino to gdino_boxes (arg 7-9) and leave known_boxes empty.
        return hybrid_match_and_fuse(
            args[0], args[1], args[2],
            None, None, None, # empty known
            args[3], args[4], args[5], # gdino
            **kwargs
        )
    return hybrid_match_and_fuse(*args, **kwargs)

class OTFusionModule:
    """Wrapper maintaining interface."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def fuse(self, *args, **kwargs):
        # Merge kwargs
        params = {**self.kwargs, **kwargs}
        return ot_fusion(*args, **params)
