"""
Adaptive Proposal Fusion (APF) Module
======================================
Replaces naive concatenation + OT Fusion with weighted box fusion (WBF).
Supports multiple proposal sources with rule-based and MLP-based weighting.

Source IDs:
  0 = Mask R-CNN known
  1 = Grounding DINO
  2 = VLRM / Stage 2

Architecture:
  Input:  Proposals from N sources (boxes, scores, classes, source_ids)
  Step 1: Per-proposal feature computation (area, aspect_ratio, density, max_iou)
  Step 2: Adaptive weight computation (rule-based or MLP)
  Step 3: Weighted Box Fusion (cluster overlapping proposals, fuse each cluster)
  Step 4: Source contribution tracking per final detection
  Output: Fused proposals + contribution map
"""

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from typing import Dict, List, Optional, Tuple, Any


class FusionMLP(torch.nn.Module):
    """
    Lightweight MLP for learned fusion weights.

    Input features (8-dim):
      [score, source_0, source_1, source_2, area_norm, aspect_log, max_iou, density]

    Output: scalar fusion weight in (0, 1) after sigmoid.
    """
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveProposalFusion:
    """
    Adaptive Proposal Fusion.

    Two weighting modes:
      rule-based (default) : hand-tuned per-source weights with modulation
      MLP-based             : learned weight via FusionMLP (requires pretrained model)

    Two fusion strategies:
      WBF (default) : Weighted Box Fusion — cluster + weighted average
      Soft-NMS      : Gaussian score decay for overlapping boxes
    """

    def __init__(
        self,
        source_weights: Optional[Dict[int, float]] = None,
        wbf_iou_threshold: float = 0.5,
        wbf_score_threshold: float = 0.05,
        wbf_max_boxes: int = 500,
        fusion_mlp: Optional[FusionMLP] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.wbf_iou_threshold = wbf_iou_threshold
        self.wbf_score_threshold = wbf_score_threshold
        self.wbf_max_boxes = wbf_max_boxes
        self.fusion_mlp = fusion_mlp
        self.device = device

        # Source base weights (tuned for LVIS)
        self.source_weights = source_weights or {0: 0.55, 1: 0.50, 2: 0.40}

        # High-confidence boost thresholds and factors
        self.boost_thresholds = {0: 0.90, 1: 0.80, 2: 0.70}
        self.boost_factors = {0: 1.18, 1: 1.20, 2: 1.375}

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------
    def _compute_features(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        source_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        N = len(boxes)
        if N == 0:
            return {}

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        aspect = widths / (heights + 1e-6)

        area_norm = areas / (areas.sum() + 1e-6)

        if N < 2:
            max_iou = torch.zeros(N, device=boxes.device, dtype=scores.dtype)
            density = torch.zeros(N, device=boxes.device, dtype=scores.dtype)
        else:
            iou_mat = box_iou(boxes, boxes)
            max_iou = iou_mat.topk(2, dim=1)[0][:, 1]
            density = (iou_mat > self.wbf_iou_threshold).float().sum(dim=1) - 1

        return {
            "areas": areas,
            "area_norm": area_norm,
            "aspect": aspect,
            "max_iou": max_iou,
            "density": density,
            "scores": scores,
            "classes": classes,
            "source_ids": source_ids,
        }

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------
    def _compute_weights(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        scores = feats["scores"]
        source_ids = feats["source_ids"]

        if self.fusion_mlp is not None:
            return self._mlp_weights(feats)

        # Rule-based weighting
        weights = torch.zeros_like(scores)
        for src_id in (0, 1, 2):
            mask = source_ids == src_id
            if not mask.any():
                continue
            base = self.source_weights.get(src_id, 0.5)
            boost_thr = self.boost_thresholds.get(src_id, 0.9)
            boost_fac = self.boost_factors.get(src_id, 1.0)
            high = scores[mask] > boost_thr
            weights[mask] = base * torch.where(high, torch.tensor(boost_fac, dtype=scores.dtype, device=scores.device), torch.ones_like(scores[mask]))
            # Area modulation (±20% for extreme sizes)
            a_norm = feats["area_norm"][mask]
            a_mod = 1.0 + 0.2 * ((a_norm - a_norm.mean()) / (a_norm.std() + 1e-6)).clamp(-2, 2).tanh()
            weights[mask] = weights[mask] * a_mod
        return weights

    def _mlp_weights(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        N = len(feats["scores"])
        src_onehot = torch.zeros(N, 3, device=self.device)
        src_onehot.scatter_(1, feats["source_ids"].unsqueeze(1).long(), 1.0)

        mlp_in = torch.cat(
            [
                feats["scores"].unsqueeze(1),
                src_onehot,
                feats["area_norm"].unsqueeze(1),
                feats["aspect"].unsqueeze(1).log1p(),
                feats["max_iou"].unsqueeze(1),
                (feats["density"] / 100.0).unsqueeze(1),
            ],
            dim=1,
        )
        with torch.no_grad():
            return torch.sigmoid(self.fusion_mlp(mlp_in).squeeze(1))

    # ------------------------------------------------------------------
    # Weighted Box Fusion (WBF)
    # ------------------------------------------------------------------
    def _wbf(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        source_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        N = len(boxes)
        if N == 0:
            return self._empty()

        # Sort by score descending for priority clustering
        order = torch.argsort(scores, descending=True)
        boxes, scores = boxes[order], scores[order]
        classes, source_ids = classes[order], source_ids[order]
        weights = weights[order]

        fused_boxes, fused_scores, fused_classes, fused_contrib = [], [], [], []
        assigned = torch.zeros(N, dtype=torch.bool, device=self.device)

        for i in range(N):
            if assigned[i]:
                continue

            # Build cluster: all unassigned boxes with IoU > threshold vs box i
            free = ~assigned
            ious = box_iou(boxes[i:i + 1], boxes[free])[0]
            cluster_mask = free.clone()
            cluster_mask[free] = ious > self.wbf_iou_threshold

            idx = torch.where(cluster_mask)[0]
            assigned[idx] = True

            c_boxes = boxes[idx]
            c_scores = scores[idx]
            c_classes = classes[idx]
            c_weights = weights[idx]
            c_src = source_ids[idx]

            # Weighted average box
            w_sum = c_weights.sum() + 1e-6
            f_box = (c_boxes * c_weights.unsqueeze(1)).sum(dim=0) / w_sum

            # Weighted average score
            f_score = (c_scores * c_weights).sum() / w_sum

            # Class by weighted voting
            votes = torch.zeros(1203, device=self.device)
            for j in range(len(idx)):
                votes[c_classes[j]] += c_weights[j] * c_scores[j]
            f_cls = votes.argmax()

            # Source contribution
            contrib = torch.zeros(3, device=self.device)
            for s in (0, 1, 2):
                sm = c_src == s
                if sm.any():
                    contrib[s] = (c_weights[sm] * c_scores[sm]).sum()
            contrib = contrib / (contrib.sum() + 1e-6)

            fused_boxes.append(f_box)
            fused_scores.append(f_score)
            fused_classes.append(f_cls)
            fused_contrib.append(contrib)

        if not fused_boxes:
            return self._empty()

        boxes = torch.stack(fused_boxes)
        scores = torch.tensor(fused_scores, device=self.device)
        classes = torch.tensor(fused_classes, dtype=torch.long, device=self.device)
        contrib = torch.stack(fused_contrib)

        # Score threshold
        keep = scores >= self.wbf_score_threshold
        boxes, scores, classes, contrib = boxes[keep], scores[keep], classes[keep], contrib[keep]

        # Re-sort and cap
        order = torch.argsort(scores, descending=True)
        if len(order) > self.wbf_max_boxes:
            order = order[:self.wbf_max_boxes]

        return {
            "boxes": boxes[order],
            "scores": scores[order],
            "classes": classes[order],
            "source_contributions": contrib[order],
        }

    # ------------------------------------------------------------------
    # Soft-NMS (alternative to WBF)
    # ------------------------------------------------------------------
    def _soft_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
        source_ids: torch.Tensor,
        weights: torch.Tensor,
        sigma: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        N = len(boxes)
        if N == 0:
            return self._empty()

        order = torch.argsort(scores, descending=True)
        boxes, scores = boxes[order], scores[order].clone()
        classes, source_ids = classes[order], source_ids[order]
        weights = weights[order]

        for i in range(N):
            if scores[i] == 0:
                continue
            ious = box_iou(boxes[i:i + 1], boxes[i + 1:])[0]
            scores[i + 1:] = scores[i + 1:] * torch.exp(-(ious ** 2) / sigma)

        keep = scores >= self.wbf_score_threshold
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        source_ids, weights = source_ids[keep], weights[keep]

        contrib = torch.zeros(len(boxes), 3, device=self.device)
        for s in (0, 1, 2):
            sm = source_ids == s
            if sm.any():
                contrib[sm, s] = weights[sm]
        contrib = contrib / (contrib.sum(dim=1, keepdim=True) + 1e-6)

        order = torch.argsort(scores, descending=True)
        if len(order) > self.wbf_max_boxes:
            order = order[:self.wbf_max_boxes]

        return {
            "boxes": boxes[order],
            "scores": scores[order],
            "classes": classes[order],
            "source_contributions": contrib[order],
        }

    def _empty(self):
        return {
            "boxes": torch.empty(0, 4, device=self.device),
            "scores": torch.empty(0, device=self.device),
            "classes": torch.empty(0, dtype=torch.long, device=self.device),
            "source_contributions": torch.empty(0, 3, device=self.device),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def fuse(
        self,
        proposals_by_source: Dict[int, Dict[str, torch.Tensor]],
        use_soft_nms: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse proposals from multiple sources.

        Args:
            proposals_by_source: {source_id: {"boxes": (N,4), "scores": (N,), "classes": (N,)}}
            use_soft_nms: If True, use Soft-NMS instead of WBF.

        Returns:
            dict with keys: boxes, scores, classes, source_contributions
        """
        if not proposals_by_source:
            return self._empty()

        all_boxes, all_scores, all_classes, all_src = [], [], [], []
        for src_id, p in proposals_by_source.items():
            n = len(p["boxes"])
            if n == 0:
                continue
            all_boxes.append(p["boxes"])
            all_scores.append(p["scores"])
            all_classes.append(p["classes"])
            all_src.append(torch.full((n,), src_id, device=self.device))

        if not all_boxes:
            return self._empty()

        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        classes = torch.cat(all_classes)
        source_ids = torch.cat(all_src)

        feats = self._compute_features(boxes, scores, classes, source_ids)
        weights = self._compute_weights(feats)

        if use_soft_nms:
            return self._soft_nms(boxes, scores, classes, source_ids, weights)
        return self._wbf(boxes, scores, classes, source_ids, weights)

    @torch.no_grad()
    def deduplicate(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        classes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Single-source deduplication via WBF (replaces batched_nms).
        All proposals treated as same source (ID=0).
        """
        return self.fuse({0: {"boxes": boxes, "scores": scores, "classes": classes}})

    def update_source_stats(
        self,
        source_contributions: torch.Tensor,
        running_stats: Optional[Dict[int, float]] = None,
    ) -> Dict[int, float]:
        """Online update of per-source contribution stats (momentum=0.9)."""
        if source_contributions.numel() == 0:
            return running_stats or self.source_weights.copy()

        avg = source_contributions.mean(dim=0)
        if running_stats is None:
            running_stats = {i: 0.0 for i in range(3)}

        momentum = 0.9
        for s in range(3):
            running_stats[s] = momentum * running_stats.get(s, 0.0) + (1 - momentum) * avg[s].item()
        return running_stats
