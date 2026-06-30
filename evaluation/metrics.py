"""
Comprehensive Metrics Module
==============================

Standalone module for computing all detection evaluation metrics.

Provides:
- VOC-style AP (11-point and all-point interpolation)
- Precision, Recall, F1
- Average IoU of matched detections
- Confusion matrix
- False Positives / False Negatives
- Inference timing statistics
"""

import numpy as np
import torch
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.

    Args:
        boxes_a: (N, 4) array in [x1, y1, x2, y2] format
        boxes_b: (M, 4) array in [x1, y1, x2, y2] format

    Returns:
        iou_matrix: (N, M) array of IoU values
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / np.maximum(union, 1e-6)


def compute_voc_ap(
    recall: np.ndarray,
    precision: np.ndarray,
    use_07_metric: bool = False,
) -> float:
    """
    Compute VOC Average Precision.

    Args:
        recall: Sorted recall values
        precision: Corresponding precision values
        use_07_metric: If True, use VOC2007 11-point interpolation.
                       If False, use VOC2010+ all-point interpolation.

    Returns:
        AP value
    """
    if use_07_metric:
        # VOC2007: 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11.0
        return ap
    else:
        # VOC2010+: All-point interpolation
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # Make precision monotonically decreasing
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Find where recall changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # Sum ΔRecall × Precision
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


def evaluate_detections_per_class(
    all_predictions: List[Dict],
    all_gt_annotations: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
    use_07_metric: bool = False,
) -> Dict:
    """
    Compute per-class AP, precision, recall, and related metrics.

    Args:
        all_predictions: List of dicts per image:
            {"image_id": str, "boxes": np.ndarray (N,4), "scores": np.ndarray (N,),
             "classes": np.ndarray (N,)}
        all_gt_annotations: List of dicts per image:
            {"image_id": str, "boxes": np.ndarray (M,4), "classes": np.ndarray (M,),
             "difficult": np.ndarray (M,)}
        num_classes: Total number of classes
        iou_threshold: IoU threshold for matching
        use_07_metric: Use VOC2007 11-point AP

    Returns:
        dict with per-class AP, precision, recall arrays, and aggregate metrics
    """
    # Organize GT and detections by class
    gt_by_class = defaultdict(list)  # class_id -> list of (image_id, box, difficult)
    det_by_class = defaultdict(list)  # class_id -> list of (image_id, box, score)
    n_gt_by_class = defaultdict(int)

    for gt in all_gt_annotations:
        img_id = gt["image_id"]
        for i in range(len(gt["boxes"])):
            cls_id = int(gt["classes"][i])
            difficult = bool(gt["difficult"][i]) if "difficult" in gt else False
            gt_by_class[cls_id].append({
                "image_id": img_id,
                "box": gt["boxes"][i],
                "difficult": difficult,
                "matched": False,
            })
            if not difficult:
                n_gt_by_class[cls_id] += 1

    for pred in all_predictions:
        img_id = pred["image_id"]
        for i in range(len(pred["boxes"])):
            cls_id = int(pred["classes"][i])
            det_by_class[cls_id].append({
                "image_id": img_id,
                "box": pred["boxes"][i],
                "score": float(pred["scores"][i]),
            })

    per_class_ap = np.zeros(num_classes)
    per_class_recall = np.zeros(num_classes)
    per_class_precision = np.zeros(num_classes)
    per_class_f1 = np.zeros(num_classes)
    per_class_pr_curves = {}

    all_tp_ious = []  # For average IoU computation
    total_fp = 0
    total_fn = 0
    total_tp = 0

    for cls_id in range(num_classes):
        # Reset matched flags
        gt_items = gt_by_class.get(cls_id, [])
        for g in gt_items:
            g["matched"] = False

        det_items = det_by_class.get(cls_id, [])
        n_gt = n_gt_by_class.get(cls_id, 0)

        if n_gt == 0 and len(det_items) == 0:
            per_class_ap[cls_id] = -1  # No data for this class
            continue

        if n_gt == 0:
            per_class_ap[cls_id] = 0.0
            total_fp += len(det_items)
            continue

        # Sort detections by score (descending)
        det_items.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(det_items))
        fp = np.zeros(len(det_items))

        # Group GT by image for efficient matching
        gt_by_image = defaultdict(list)
        for g_idx, g in enumerate(gt_items):
            gt_by_image[g["image_id"]].append((g_idx, g))

        for d_idx, det in enumerate(det_items):
            img_id = det["image_id"]
            img_gt = gt_by_image.get(img_id, [])

            if len(img_gt) == 0:
                fp[d_idx] = 1
                continue

            # Compute IoU with all GT boxes for this image and class
            det_box = np.array([det["box"]])
            gt_boxes = np.array([g[1]["box"] for g in img_gt])
            ious = compute_iou_matrix(det_box, gt_boxes)[0]

            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold:
                g_idx, g_item = img_gt[max_iou_idx]
                if not g_item["difficult"]:
                    if not g_item["matched"]:
                        tp[d_idx] = 1
                        g_item["matched"] = True
                        gt_items[g_idx]["matched"] = True
                        all_tp_ious.append(max_iou)
                    else:
                        fp[d_idx] = 1
                # If difficult, neither TP nor FP
            else:
                fp[d_idx] = 1

        # Compute precision-recall curve
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        recall_curve = cum_tp / n_gt
        precision_curve = cum_tp / (cum_tp + cum_fp)

        # Store PR curve
        per_class_pr_curves[cls_id] = {
            "recall": recall_curve,
            "precision": precision_curve,
            "scores": np.array([d["score"] for d in det_items]),
        }

        # Compute AP
        ap = compute_voc_ap(recall_curve, precision_curve, use_07_metric)
        per_class_ap[cls_id] = ap

        # Final precision/recall at all detections
        if len(recall_curve) > 0:
            per_class_recall[cls_id] = recall_curve[-1]
            # Use precision at the point of maximum F1
            f1_curve = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-6)
            best_idx = np.argmax(f1_curve)
            per_class_precision[cls_id] = precision_curve[best_idx]
            per_class_f1[cls_id] = f1_curve[best_idx]

        # Count FP/FN for this class
        class_tp = int(cum_tp[-1]) if len(cum_tp) > 0 else 0
        class_fp = int(cum_fp[-1]) if len(cum_fp) > 0 else 0
        class_fn = n_gt - class_tp

        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn

    # Compute aggregate metrics (only for classes with GT)
    valid_mask = per_class_ap >= 0
    mAP = np.mean(per_class_ap[valid_mask]) if valid_mask.any() else 0.0

    avg_iou = np.mean(all_tp_ious) if all_tp_ious else 0.0

    return {
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "per_class_pr_curves": per_class_pr_curves,
        "mAP": mAP,
        "avg_iou": avg_iou,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def compute_ap_at_iou(
    all_predictions: List[Dict],
    all_gt_annotations: List[Dict],
    num_classes: int,
    iou_threshold: float,
    use_07_metric: bool = False,
) -> float:
    """Compute mAP at a specific IoU threshold."""
    result = evaluate_detections_per_class(
        all_predictions, all_gt_annotations, num_classes,
        iou_threshold=iou_threshold, use_07_metric=use_07_metric,
    )
    return result["mAP"]


def compute_precision_recall_f1(
    all_predictions: List[Dict],
    all_gt_annotations: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> Dict:
    """
    Compute overall Precision, Recall, F1 at a given score threshold.

    Returns:
        dict with precision, recall, f1
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, gt in zip(all_predictions, all_gt_annotations):
        # Filter predictions by score threshold
        if len(pred["scores"]) > 0:
            mask = pred["scores"] >= score_threshold
            pred_boxes = pred["boxes"][mask]
            pred_classes = pred["classes"][mask]
        else:
            pred_boxes = np.empty((0, 4))
            pred_classes = np.empty(0, dtype=int)

        gt_boxes = gt["boxes"]
        gt_classes = gt["classes"]
        gt_matched = np.zeros(len(gt_boxes), dtype=bool)

        for i in range(len(pred_boxes)):
            # Find best matching GT box with same class
            best_iou = 0
            best_j = -1
            for j in range(len(gt_boxes)):
                if gt_matched[j] or gt_classes[j] != pred_classes[i]:
                    continue
                iou = compute_iou_matrix(
                    pred_boxes[i:i+1], gt_boxes[j:j+1]
                )[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_threshold and best_j >= 0:
                total_tp += 1
                gt_matched[best_j] = True
            else:
                total_fp += 1

        total_fn += np.sum(~gt_matched)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


class InferenceTimer:
    """Tracks inference time and computes FPS / GPU memory statistics."""

    def __init__(self):
        self.times = []
        self.gpu_memory_peak = 0
        self._start = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start
        self.times.append(elapsed)

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / 1e9
            self.gpu_memory_peak = max(self.gpu_memory_peak, mem)

    def get_stats(self) -> Dict:
        if len(self.times) == 0:
            return {"total_time": 0, "avg_time": 0, "fps": 0, "gpu_memory_gb": 0}

        total = sum(self.times)
        avg = total / len(self.times)
        fps = len(self.times) / total if total > 0 else 0

        return {
            "total_time": total,
            "avg_time_per_image": avg,
            "fps": fps,
            "num_images": len(self.times),
            "gpu_memory_peak_gb": self.gpu_memory_peak,
        }
