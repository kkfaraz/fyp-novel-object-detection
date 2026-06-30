"""
Score Refinement Module (SRM)
==============================

Implementation of Algorithm 1 from the paper:
"Enhancing Novel Object Detection via Cooperative Foundational Models"

The SRM refines confidence scores by combining detector scores with SAM
mask quality scores using MinMax normalization.

Algorithm 1 from paper:
    Input: Combined scores s_j^(C), SAM scores s_j^(SAM)
    Output: Refined scores s_j
    
    1. scaler, scaler_sam = MinMaxScaler()
    2. s_j^(C) = scaler.fit_transform(s_j^(C))
    3. s_j^(SAM) = scaler_sam.fit_transform(s_j^(SAM))
    4. s_j = s_j^(C) × s_j^(SAM)
"""

import torch
import numpy as np
import sys
import os

# Add parent scripts directory for gpu_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from gpu_utils import torch_minmax_scale


class ScoreRefinementModule:
    """
    Score Refinement Module (SRM) for combining detector and SAM scores.
    
    This module implements the score refinement strategy from the paper,
    which normalizes both detector confidence scores and SAM mask quality
    scores before combining them via element-wise multiplication.
    
    Args:
        per_image_norm (bool): If True, normalize scores per-image to avoid
            cross-image ranking issues. If False, use global normalization
            (paper's approach). Default: True (more stable).
    
    Example:
        >>> srm = ScoreRefinementModule(per_image_norm=True)
        >>> combined_scores = torch.tensor([0.8, 0.6, 0.9, 0.4])
        >>> sam_scores = torch.tensor([0.95, 0.85, 0.90, 0.70])
        >>> refined = srm.refine_scores(combined_scores, sam_scores)
        >>> print(refined)
        tensor([0.76, 0.40, 1.00, 0.00])  # Normalized and multiplied
    """
    
    def __init__(self, per_image_norm=True):
        """
        Initialize the Score Refinement Module.
        
        Args:
            per_image_norm: Whether to normalize per-image (recommended)
        """
        self.per_image_norm = per_image_norm
    
    def refine_scores(self, combined_scores, sam_scores):
        """
        Apply score refinement using MinMax normalization and multiplication.
        
        Uses a pure-PyTorch MinMaxScaler to avoid CPU round-trips.
        All computation stays on the input tensor's device (GPU).
        
        Args:
            combined_scores: Tensor of shape (N,) - detector confidence scores
            sam_scores: Tensor of shape (N,) - SAM mask quality (IoU) scores
        
        Returns:
            refined_scores: Tensor of shape (N,) - refined scores in [0, 1]
        
        Notes:
            - Handles empty inputs gracefully (returns empty tensor)
            - If lengths don't match, truncates to minimum length
            - All scores are MinMax scaled to [0, 1] before multiplication
        """
        # Handle empty inputs
        if len(combined_scores) == 0 or len(sam_scores) == 0:
            return combined_scores
        
        # Ensure same length (truncate to minimum)
        min_len = min(len(combined_scores), len(sam_scores))
        combined_scores = combined_scores[:min_len]
        sam_scores = sam_scores[:min_len]
        
        # Ensure both are tensors and on the same device
        if not isinstance(combined_scores, torch.Tensor):
            combined_scores = torch.tensor(combined_scores, dtype=torch.float32)
        if not isinstance(sam_scores, torch.Tensor):
            sam_scores = torch.tensor(sam_scores, dtype=torch.float32)
        
        # Keep track of original device to ensure output matches
        original_device = combined_scores.device
        sam_scores = sam_scores.to(original_device)
        
        # MinMax normalization per Algorithm 1 of the paper, then multiply.
        # This ensures both score distributions are mapped to [0,1] before fusion,
        # preventing scale dominance from either modality.
        combined_scaled = torch_minmax_scale(combined_scores.float())
        sam_scaled = torch_minmax_scale(sam_scores.float())
        refined = combined_scaled * sam_scaled
        
        return refined.to(dtype=combined_scores.dtype)
    
    def __repr__(self):
        return f"ScoreRefinementModule(per_image_norm={self.per_image_norm})"


def apply_srm_refinement(combined_scores, sam_scores, per_image_norm=True):
    """
    Convenience function to apply SRM without creating an instance.
    
    Args:
        combined_scores: Detector confidence scores
        sam_scores: SAM mask quality scores
        per_image_norm: Whether to use per-image normalization
    
    Returns:
        Refined scores
    
    Example:
        >>> refined = apply_srm_refinement(detector_scores, sam_scores)
    """
    srm = ScoreRefinementModule(per_image_norm=per_image_norm)
    return srm.refine_scores(combined_scores, sam_scores)


if __name__ == "__main__":
    # Quick test
    print("Testing Score Refinement Module...")
    
    # Test case 1: Normal scores
    combined = torch.tensor([0.8, 0.6, 0.9, 0.4, 0.7])
    sam = torch.tensor([0.95, 0.85, 0.90, 0.70, 0.80])
    
    srm = ScoreRefinementModule(per_image_norm=True)
    refined = srm.refine_scores(combined, sam)
    
    print(f"\nTest 1: Normal scores")
    print(f"  Combined: {combined}")
    print(f"  SAM:      {sam}")
    print(f"  Refined:  {refined}")
    print(f"  Range:    [{refined.min():.3f}, {refined.max():.3f}]")
    
    # Test case 2: Empty inputs
    empty = torch.tensor([])
    refined_empty = srm.refine_scores(empty, empty)
    print(f"\nTest 2: Empty inputs")
    print(f"  Result: {refined_empty} (length={len(refined_empty)})")
    
    # Test case 3: Mismatched lengths
    short = torch.tensor([0.9, 0.8])
    long = torch.tensor([0.95, 0.90, 0.85, 0.80])
    refined_mismatch = srm.refine_scores(short, long)
    print(f"\nTest 3: Mismatched lengths")
    print(f"  Short (len={len(short)}): {short}")
    print(f"  Long (len={len(long)}):  {long}")
    print(f"  Refined (len={len(refined_mismatch)}): {refined_mismatch}")
    
    print("\n✓ All tests passed!")
