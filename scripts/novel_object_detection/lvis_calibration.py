"""
LVIS Class-Frequency-Aware Score Calibration
=============================================

Adjusts detection scores based on LVIS category frequency groups
(rare/common/frequent) to compensate for the natural suppression
of rare classes by CLIP/VLRM in large-vocabulary settings.

This is a dataset-adaptation module, NOT a pipeline redesign.
The underlying detection pipeline remains identical to COCO.

LVIS Frequency Groups:
  - Rare (r):     categories with 1-10 training instances  (~454 classes)
  - Common (c):   categories with 11-100 instances         (~461 classes)  
  - Frequent (f): categories with >100 instances            (~288 classes)
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from detectron2.data import MetadataCatalog


def get_lvis_frequency_groups(data_split: str = "lvis_v1_val") -> Dict[str, List[int]]:
    """
    Get LVIS category frequency group assignments.
    
    Uses the LVIS API metadata to determine which classes are rare/common/frequent.
    Falls back to the LVIS annotation file if metadata is not directly available.
    
    Args:
        data_split: Detectron2 dataset name (e.g., 'lvis_v1_val')
    
    Returns:
        Dict with keys 'rare', 'common', 'frequent', each mapping to a list of 
        0-indexed class IDs belonging to that frequency group.
    """
    try:
        meta = MetadataCatalog.get(data_split)
        
        # Try to get frequency info from LVIS API
        json_file = meta.json_file
        from lvis import LVIS
        lvis_api = LVIS(json_file)
        
        freq_groups = {"rare": [], "common": [], "frequent": []}
        class_counts = torch.zeros(1203)
        
        for cat in lvis_api.cats.values():
            # LVIS categories are 1-indexed, convert to 0-indexed
            cat_idx = cat["id"] - 1
            freq = cat.get("frequency", "")
            count = cat.get("image_count", 0)
            if cat_idx < 1203:
                class_counts[cat_idx] = count
            
            if freq == "r":
                freq_groups["rare"].append(cat_idx)
            elif freq == "c":
                freq_groups["common"].append(cat_idx)
            elif freq == "f":
                freq_groups["frequent"].append(cat_idx)
            else:
                # Unknown frequency, treat as common
                freq_groups["common"].append(cat_idx)
        
        print(f"[LVIS Calibration] Frequency groups loaded: "
              f"rare={len(freq_groups['rare'])}, "
              f"common={len(freq_groups['common'])}, "
              f"frequent={len(freq_groups['frequent'])}")
        
        return freq_groups, class_counts
        
    except Exception as e:
        print(f"[LVIS Calibration] Warning: Could not load frequency groups: {e}")
        print(f"[LVIS Calibration] Falling back to no frequency calibration")
        return {"rare": [], "common": [], "frequent": []}, torch.zeros(1203)


def build_frequency_boost_map(
    num_classes: int,
    freq_groups: Dict[str, List[int]],
    rare_boost: float = 1.5,
    common_boost: float = 1.15,
    frequent_boost: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Build a per-class multiplicative boost tensor.
    
    Args:
        num_classes: Total number of classes (1203 for LVIS v1)
        freq_groups: Output of get_lvis_frequency_groups()
        rare_boost: Multiplicative factor for rare classes
        common_boost: Multiplicative factor for common classes
        frequent_boost: Multiplicative factor for frequent classes
        device: Device to create tensor on (default: CPU)
    
    Returns:
        boost_map: Tensor of shape (num_classes,) with per-class boost factors
    """
    boost_map = torch.ones(num_classes, device=device)
    
    for idx in freq_groups.get("rare", []):
        if idx < num_classes:
            boost_map[idx] = rare_boost
    
    for idx in freq_groups.get("common", []):
        if idx < num_classes:
            boost_map[idx] = common_boost
    
    for idx in freq_groups.get("frequent", []):
        if idx < num_classes:
            boost_map[idx] = frequent_boost
    
    return boost_map


def apply_frequency_calibration(
    scores: torch.Tensor,
    labels: torch.Tensor,
    boost_map: torch.Tensor
) -> torch.Tensor:
    """
    Apply per-class frequency-aware score calibration.
    
    Multiplies each detection's score by its class's frequency boost factor.
    This compensates for the natural suppression of rare classes in CLIP/VLRM.
    
    Args:
        scores: Detection scores, shape (N,)
        labels: Class labels, shape (N,) — 0-indexed
        boost_map: Per-class boost factors from build_frequency_boost_map()
    
    Returns:
        calibrated_scores: Adjusted scores, shape (N,)
    """
    if len(scores) == 0:
        return scores
    
    # Gather the boost factor for each detection's class
    labels_clamped = labels.clamp(0, len(boost_map) - 1).long()
    boosts = boost_map[labels_clamped]
    
    calibrated = scores * boosts.to(scores.device)
    
    # NOTE: Do NOT clamp to [0,1] — calibrated scores are used for relative
    # ranking in NMS/Top-K only. Clamping destroys ranking for boosted classes
    # (e.g. rare score 0.6 * 1.5 = 0.9, but 0.7 * 1.5 = 1.0 → same rank).
    
    return calibrated


class LVISCalibrator:
    """
    Stateful calibrator that loads frequency groups once and applies calibration
    to any batch of detections.
    
    Usage:
        calibrator = LVISCalibrator(data_split="lvis_v1_val")
        calibrated_scores = calibrator.calibrate(scores, labels)
    """
    
    def __init__(
        self,
        data_split: str = "lvis_v1_val",
        num_classes: int = 1203,
        rare_boost: float = 1.5,
        common_boost: float = 1.15,
        frequent_boost: float = 1.0,
        device: torch.device = None
    ):
        self.data_split = data_split
        self.num_classes = num_classes
        self.device = device
        
        # Load frequency groups and build boost map on target device
        self.freq_groups, self.class_counts = get_lvis_frequency_groups(data_split)
        self.class_counts = self.class_counts.to(device)
        self.boost_map = build_frequency_boost_map(
            num_classes, self.freq_groups,
            rare_boost=rare_boost,
            common_boost=common_boost,
            frequent_boost=frequent_boost,
            device=device
        )
        
        # Logit Adjustment priors
        total_count = self.class_counts.sum()
        self.priors = (self.class_counts + 1e-5) / (total_count + 1e-5)
        
        # Stats tracking
        self._calibrated_count = 0
        self._rare_detections = 0
        self._common_detections = 0
        self._frequent_detections = 0
        
        print(f"[LVIS Calibrator] Initialized: rare×{rare_boost}, "
              f"common×{common_boost}, frequent×{frequent_boost}, device={device}")
    
    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor, apply_logit_adjustment: bool = True, tau: float = 0.5) -> torch.Tensor:
        """Apply frequency-aware calibration and logit adjustment to a batch of detections.
        
        Note: tau=0.5 (reduced from 1.0) to prevent double-calibration.
        Logit adjustment alone adds +τ×log(1/prior) which for rare classes with few images
        is already a large boost. Combined with multiplicative frequency boost (rare×1.5),
        tau=1.0 was causing false positives for rare categories to rank above true positives
        for common ones. tau=0.5 provides a gentler correction.
        """
        
        # Mod 6: Logit Adjustment (converted to logit space for correctness)
        if apply_logit_adjustment and len(labels) > 0:
            labels_clamped = labels.clamp(0, self.num_classes - 1).long()
            priors = self.priors[labels_clamped].to(scores.device)
            # Convert probability scores to logit space, adjust, then sigmoid back
            scores = torch.clamp(scores, 1e-7, 1 - 1e-7)
            logits = torch.logit(scores)
            logits = logits - tau * torch.log(priors)
            scores = torch.sigmoid(logits)
            
        calibrated = apply_frequency_calibration(scores, labels, self.boost_map)
        
        # Track stats
        if len(labels) > 0:
            self._calibrated_count += len(labels)
            rare_set = set(self.freq_groups.get("rare", []))
            common_set = set(self.freq_groups.get("common", []))
            for l in labels.tolist():
                if l in rare_set:
                    self._rare_detections += 1
                elif l in common_set:
                    self._common_detections += 1
                else:
                    self._frequent_detections += 1
        
        return calibrated
    
    def print_stats(self):
        """Print calibration statistics."""
        print(f"\n[LVIS Calibrator Stats]")
        print(f"  Total calibrated: {self._calibrated_count}")
        print(f"  Rare detections:     {self._rare_detections}")
        print(f"  Common detections:   {self._common_detections}")
        print(f"  Frequent detections: {self._frequent_detections}")
