"""
VOC Evaluator
==============

Custom evaluator for PASCAL VOC that computes all requested metrics
with known/novel class split support.

Mirrors the interface of the existing CustomEvaluator / LVISEvaluatorCustom
to maintain compatibility with the 3-stage pipeline.
"""

import numpy as np
import torch
import json
import os
import time
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict, defaultdict

from detectron2.evaluation.coco_evaluation import instances_to_coco_json

from .metrics import (
    evaluate_detections_per_class,
    compute_ap_at_iou,
    compute_precision_recall_f1,
    InferenceTimer,
)


class VOCEvaluator:
    """
    PASCAL VOC evaluator with known/novel class split support.

    Compatible with the pipeline's evaluator.process() / evaluator.evaluate() pattern.

    Args:
        class_names: List of class names (e.g., VOC_CLASSES)
        known_class_ids: List of 0-indexed known class IDs
        novel_class_ids: List of 0-indexed novel class IDs
        output_dir: Directory to save results
        gt_annotations: List of dicts with ground truth per image
        use_07_metric: Use VOC2007 11-point AP interpolation
    """

    def __init__(
        self,
        class_names: List[str],
        known_class_ids: List[int],
        novel_class_ids: List[int],
        output_dir: str = "results/voc_baseline",
        gt_annotations: Optional[List[Dict]] = None,
        use_07_metric: bool = False,
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.known_class_ids = known_class_ids
        self.novel_class_ids = novel_class_ids
        self.output_dir = output_dir
        self.use_07_metric = use_07_metric

        # Store ground truth indexed by image_id
        self._gt_by_image = {}
        if gt_annotations is not None:
            for gt in gt_annotations:
                self._gt_by_image[gt["image_id"]] = gt

        self._predictions = []
        self._gt_list = []
        self.timer = InferenceTimer()

        os.makedirs(output_dir, exist_ok=True)

    def reset(self):
        """Reset accumulated predictions."""
        self._predictions = []
        self._gt_list = []
        self.timer = InferenceTimer()

    def set_gt_annotations(self, gt_annotations: List[Dict]):
        """Set ground truth annotations (can be called after init)."""
        self._gt_by_image = {}
        for gt in gt_annotations:
            self._gt_by_image[gt["image_id"]] = gt

    def process(self, inputs: List[Dict], outputs: List[Dict]):
        """
        Process a batch of predictions.

        Matches the interface of CustomEvaluator.process() from the original pipeline.

        Args:
            inputs: List of input dicts with "image_id", "file_name", etc.
            outputs: List of output dicts with "instances" (detectron2 Instances)
        """
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            instances = out["instances"].to("cpu")

            # Convert instances to numpy arrays
            if len(instances) > 0:
                boxes = instances.pred_boxes.tensor.numpy()
                scores = instances.scores.numpy()
                classes = instances.pred_classes.numpy()
            else:
                boxes = np.empty((0, 4))
                scores = np.empty(0)
                classes = np.empty(0, dtype=int)

            self._predictions.append({
                "image_id": image_id,
                "boxes": boxes,
                "scores": scores,
                "classes": classes,
            })

            # Build GT entry for this image
            if image_id in self._gt_by_image:
                gt = self._gt_by_image[image_id]
                gt_boxes = np.array([ann["bbox"] for ann in gt.get("annotations", [])])
                gt_classes = np.array([ann["category_id"] for ann in gt.get("annotations", [])])
                gt_difficult = np.array([
                    ann.get("difficult", 0) for ann in gt.get("annotations", [])
                ])
            else:
                gt_boxes = np.empty((0, 4))
                gt_classes = np.empty(0, dtype=int)
                gt_difficult = np.empty(0, dtype=int)

            self._gt_list.append({
                "image_id": image_id,
                "boxes": gt_boxes,
                "classes": gt_classes,
                "difficult": gt_difficult,
            })

    def evaluate(self) -> Dict:
        """
        Compute all metrics and return results.

        Returns:
            dict with all metrics including Known AP, Novel AP, mAP, etc.
        """
        print("\n" + "=" * 70)
        print("VOC Evaluation — Computing Metrics")
        print("=" * 70)

        if len(self._predictions) == 0:
            print("[VOC Evaluator] No predictions to evaluate!")
            return {}

        # ================================================================
        # AP at IoU=0.5 (standard VOC metric)
        # ================================================================
        results_50 = evaluate_detections_per_class(
            self._predictions, self._gt_list, self.num_classes,
            iou_threshold=0.5, use_07_metric=self.use_07_metric,
        )

        # ================================================================
        # AP at IoU=0.75
        # ================================================================
        results_75 = evaluate_detections_per_class(
            self._predictions, self._gt_list, self.num_classes,
            iou_threshold=0.75, use_07_metric=self.use_07_metric,
        )

        # ================================================================
        # AP at IoU=0.5:0.95 (COCO-style mAP)
        # ================================================================
        ap_at_ious = []
        for iou_thr in np.arange(0.5, 1.0, 0.05):
            ap = compute_ap_at_iou(
                self._predictions, self._gt_list, self.num_classes,
                iou_threshold=iou_thr, use_07_metric=self.use_07_metric,
            )
            ap_at_ious.append(ap)
        mAP_50_95 = np.mean(ap_at_ious)

        # ================================================================
        # Known / Novel split metrics
        # ================================================================
        per_class_ap_50 = results_50["per_class_ap"]
        per_class_recall_50 = results_50["per_class_recall"]

        # Filter out classes with no GT (AP == -1)
        def _mean_for_ids(arr, ids):
            vals = [arr[i] for i in ids if arr[i] >= 0]
            return np.mean(vals) if vals else 0.0

        known_ap = _mean_for_ids(per_class_ap_50, self.known_class_ids)
        novel_ap = _mean_for_ids(per_class_ap_50, self.novel_class_ids)
        known_recall = _mean_for_ids(per_class_recall_50, self.known_class_ids)
        novel_recall = _mean_for_ids(per_class_recall_50, self.novel_class_ids)

        # ================================================================
        # Precision / Recall / F1 at optimal threshold
        # ================================================================
        prf = compute_precision_recall_f1(
            self._predictions, self._gt_list, self.num_classes,
            iou_threshold=0.5, score_threshold=0.3,
        )

        # ================================================================
        # Inference statistics
        # ================================================================
        timing = self.timer.get_stats()

        # ================================================================
        # Compile results
        # ================================================================
        results = OrderedDict()

        # Primary metrics
        results["mAP"] = float(results_50["mAP"] * 100)
        results["AP50"] = float(results_50["mAP"] * 100)
        results["AP75"] = float(results_75["mAP"] * 100)
        results["mAP_50_95"] = float(mAP_50_95 * 100)

        # Known / Novel
        results["Known_AP"] = float(known_ap * 100)
        results["Novel_AP"] = float(novel_ap * 100)
        results["Known_Recall"] = float(known_recall * 100)
        results["Novel_Recall"] = float(novel_recall * 100)

        # Detection quality
        results["Recall"] = float(prf["recall"] * 100)
        results["Precision"] = float(prf["precision"] * 100)
        results["F1"] = float(prf["f1"] * 100)
        results["Average_IoU"] = float(results_50["avg_iou"] * 100)

        # Error analysis
        results["False_Positives"] = results_50["total_fp"]
        results["False_Negatives"] = results_50["total_fn"]

        # Timing
        results["Inference_Time_s"] = timing.get("total_time", 0)
        results["FPS"] = timing.get("fps", 0)
        results["GPU_Memory_GB"] = timing.get("gpu_memory_peak_gb", 0)

        # Per-class AP (for visualizations)
        results["per_class_ap"] = {
            self.class_names[i]: float(per_class_ap_50[i] * 100)
            for i in range(self.num_classes)
            if per_class_ap_50[i] >= 0
        }
        results["per_class_recall"] = {
            self.class_names[i]: float(per_class_recall_50[i] * 100)
            for i in range(self.num_classes)
            if per_class_recall_50[i] >= 0
        }

        # PR curves (for visualization module)
        results["_pr_curves"] = results_50["per_class_pr_curves"]
        results["_predictions"] = self._predictions
        results["_gt_list"] = self._gt_list

        # ================================================================
        # Print publication-quality tables
        # ================================================================
        self._print_results(results)

        # ================================================================
        # Save results to disk
        # ================================================================
        self._save_results(results)

        return results

    def _print_results(self, results: Dict):
        """Print publication-quality result tables."""
        num_known = len(self.known_class_ids)
        num_novel = len(self.novel_class_ids)

        print("\n" + "=" * 70)
        print("VOC OVD Evaluation Results — Cooperative Foundational Models")
        print("=" * 70)

        # COCO-style line-by-line
        print(f"\n Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all ] = {results['mAP_50_95']:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50      | area=   all ] = {results['AP50']:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.75      | area=   all ] = {results['AP75']:.3f}")

        # Summary table
        print("\n" + "=" * 70)
        print(f"Summary ({self.num_classes} classes: {num_known} known, {num_novel} novel)")
        print("=" * 70)

        print("\n| Metric | Value |")
        print("|:-------|------:|")
        print(f"| Known AP | {results['Known_AP']:.2f} |")
        print(f"| Novel AP | {results['Novel_AP']:.2f} |")
        print(f"| mAP (AP50) | {results['mAP']:.2f} |")
        print(f"| AP75 | {results['AP75']:.2f} |")
        print(f"| mAP (AP50:95) | {results['mAP_50_95']:.2f} |")
        print(f"| Recall | {results['Recall']:.2f} |")
        print(f"| Precision | {results['Precision']:.2f} |")
        print(f"| F1 | {results['F1']:.2f} |")
        print(f"| Known Recall | {results['Known_Recall']:.2f} |")
        print(f"| Novel Recall | {results['Novel_Recall']:.2f} |")
        print(f"| Average IoU | {results['Average_IoU']:.2f} |")
        print(f"| False Positives | {results['False_Positives']} |")
        print(f"| False Negatives | {results['False_Negatives']} |")
        if results['FPS'] > 0:
            print(f"| FPS | {results['FPS']:.2f} |")
            print(f"| GPU Memory (GB) | {results['GPU_Memory_GB']:.2f} |")

        # Per-class AP table
        print("\n" + "=" * 70)
        print("Per-Class AP (IoU=0.50)")
        print("=" * 70)
        print("\n| Class | AP | Type |")
        print("|:------|---:|:-----|")
        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            ap_val = results["per_class_ap"].get(cls_name, -1)
            cls_type = "Known" if cls_id in self.known_class_ids else "Novel"
            if ap_val >= 0:
                print(f"| {cls_name} | {ap_val:.2f} | {cls_type} |")
            else:
                print(f"| {cls_name} | N/A | {cls_type} |")

        # Copypaste summary
        print("\n" + "=" * 60)
        print("Results Summary")
        print("=" * 60)
        print("copypaste: Known_AP,Novel_AP,mAP,AP75,Recall,Precision,F1")
        print("copypaste: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            results["Known_AP"], results["Novel_AP"], results["mAP"],
            results["AP75"], results["Recall"], results["Precision"], results["F1"],
        ))

    def _save_results(self, results: Dict):
        """Save results to JSON file."""
        # Remove non-serializable items
        save_results = {
            k: v for k, v in results.items()
            if not k.startswith("_")
        }

        results_path = os.path.join(self.output_dir, "evaluation_results.json")
        try:
            with open(results_path, "w") as f:
                json.dump(save_results, f, indent=2, default=str)
            print(f"\n[VOC Evaluator] Results saved to {results_path}")
        except Exception as e:
            print(f"[VOC Evaluator] Warning: Could not save results: {e}")
