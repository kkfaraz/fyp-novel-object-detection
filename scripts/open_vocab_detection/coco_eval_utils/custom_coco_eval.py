# Copyright (c) Facebook, Inc. and its affiliates.
"""
FYP: Custom COCO Evaluator with Seen/Unseen Class Metrics
==========================================================

Matches the LVIS evaluation format by displaying:
- Overall metrics (AP, AP50, AP75, APs, APm, APl)
- Seen classes metrics (known classes used in training)
- Unseen classes metrics (novel classes not used in training)
"""
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from .coco_ovd_split import categories_seen, categories_unseen


class CustomCOCOEvaluator(COCOEvaluator):
    """
    COCO Evaluator with LVIS-style seen/unseen class metrics.
    """
    
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Compute mAP for all classes, seen classes, and unseen classes.
        Matches LVIS evaluation output format with detailed COCO-style metrics.
        """
        
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # ======================================================================
        # Detailed COCO-Style Metrics Output
        # ======================================================================
        stats = coco_eval.stats
        
        print("\n" + "="*70)
        print("COCO OVD Evaluation Results - Novel Object Detection Pipeline")
        print("="*70 + "\n")
        
        # Average Precision metrics
        print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[0]:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[1]:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[2]:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[3]:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[4]:.3f}")
        print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[5]:.3f}")
        
        # Average Recall metrics
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {stats[6]:.3f}")
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {stats[7]:.3f}")
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}")
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {stats[9]:.3f}")
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {stats[10]:.3f}")
        print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {stats[11]:.3f}")
        
        # ======================================================================
        # Overall Results (All Classes)
        # ======================================================================
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        
        self._logger.info("\n" + "="*70)
        self._logger.info("Evaluation results for {} (ALL CLASSES):".format(iou_type))
        self._logger.info("="*70)
        self._logger.info("\n" + create_small_table(results))

        if class_names is None or len(class_names) <= 1:
            return results

        # ======================================================================
        # Per-Category Computation
        # ======================================================================
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        seen_names = set([x['name'] for x in categories_seen])
        unseen_names = set([x['name'] for x in categories_unseen])
        
        # Store per-category results
        results_seen = {"AP": [], "AP50": [], "AP75": []}
        results_unseen = {"AP": [], "AP50": [], "AP75": []}
        
        for idx, name in enumerate(class_names):
            # AP (IoU 0.5:0.95)
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            
            # AP50 (IoU 0.5)
            precision50 = precisions[0, :, idx, 0, -1]
            precision50 = precision50[precision50 > -1]
            ap50 = np.mean(precision50) if precision50.size else float("nan")
            
            # AP75 (IoU 0.75)
            precision75 = precisions[5, :, idx, 0, -1]  # IoU=0.75 is at index 5
            precision75 = precision75[precision75 > -1]
            ap75 = np.mean(precision75) if precision75.size else float("nan")
            
            if name in seen_names:
                results_seen["AP"].append(ap * 100)
                results_seen["AP50"].append(ap50 * 100)
                results_seen["AP75"].append(ap75 * 100)
            elif name in unseen_names:
                results_unseen["AP"].append(ap * 100)
                results_unseen["AP50"].append(ap50 * 100)
                results_unseen["AP75"].append(ap75 * 100)

        # ======================================================================
        # Seen Classes Results (matching LVIS "known classes only" format)
        # ======================================================================
        seen_metrics = {}
        for metric in ["AP", "AP50", "AP75"]:
            vals = [v for v in results_seen[metric] if not np.isnan(v)]
            seen_metrics[metric] = np.mean(vals) if vals else float("nan")
        
        self._logger.info("\n" + "="*70)
        self._logger.info("Evaluation results for {} (SEEN CLASSES ONLY - {} classes):".format(
            iou_type, len(categories_seen)))
        self._logger.info("="*70)
        self._logger.info("\n" + create_small_table(seen_metrics))
        
        # ======================================================================
        # Unseen Classes Results (matching LVIS "novel classes only" format)
        # ======================================================================
        unseen_metrics = {}
        for metric in ["AP", "AP50", "AP75"]:
            vals = [v for v in results_unseen[metric] if not np.isnan(v)]
            unseen_metrics[metric] = np.mean(vals) if vals else float("nan")
        
        self._logger.info("\n" + "="*70)
        self._logger.info("Evaluation results for {} (UNSEEN/NOVEL CLASSES ONLY - {} classes):".format(
            iou_type, len(categories_unseen)))
        self._logger.info("="*70)
        self._logger.info("\n" + create_small_table(unseen_metrics))

        # ======================================================================
        # Summary Table (LVIS-style)
        # ======================================================================
        summary_data = [
            ["All Classes", f"{results['AP']:.2f}", f"{results['AP50']:.2f}", f"{results['AP75']:.2f}"],
            ["Seen Classes", f"{seen_metrics['AP']:.2f}", f"{seen_metrics['AP50']:.2f}", f"{seen_metrics['AP75']:.2f}"],
            ["Unseen/Novel Classes", f"{unseen_metrics['AP']:.2f}", f"{unseen_metrics['AP50']:.2f}", f"{unseen_metrics['AP75']:.2f}"],
        ]
        
        self._logger.info("\n" + "="*70)
        self._logger.info("SUMMARY (LVIS-style format):")
        self._logger.info("="*70)
        summary_table = tabulate(
            summary_data,
            headers=["Category Split", "AP", "AP50", "AP75"],
            tablefmt="pipe",
            numalign="right",
        )
        self._logger.info("\n" + summary_table)
        self._logger.info("="*70 + "\n")

        # Add seen/unseen metrics to results
        results["AP-seen"] = seen_metrics["AP"]
        results["AP50-seen"] = seen_metrics["AP50"]
        results["AP75-seen"] = seen_metrics["AP75"]
        results["AP-unseen"] = unseen_metrics["AP"]
        results["AP50-unseen"] = unseen_metrics["AP50"]
        results["AP75-unseen"] = unseen_metrics["AP75"]
        
        # ======================================================================
        # LVIS-Style Detailed Markdown Tables (matching LVIS format exactly)
        # ======================================================================
        num_seen = len(categories_seen)
        num_unseen = len(categories_unseen)
        num_all = num_seen + num_unseen
        
        print("\n" + "="*70)
        print(f"Evaluation results for bbox (ALL CLASSES - {num_all} categories):")
        print("="*70)
        print("\n|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |")
        print("|:------:|:------:|:------:|:-----:|:------:|:------:|")
        print(f"| {results['AP']:.3f} | {results['AP50']:.3f} | {results['AP75']:.3f} | {results['APs']:.3f} | {results['APm']:.3f} | {results['APl']:.3f} |")
        
        print("\n" + "="*70)
        print(f"Evaluation results for bbox (SEEN CLASSES - {num_seen} categories):")
        print("="*70)
        print("\n|   AP   |  AP50  |  AP75  |")
        print("|:------:|:------:|:------:|")
        print(f"| {seen_metrics['AP']:.3f} | {seen_metrics['AP50']:.3f} | {seen_metrics['AP75']:.3f} |")
        
        print("\n" + "="*70)
        print(f"Evaluation results for bbox (UNSEEN/NOVEL CLASSES - {num_unseen} categories):")
        print("="*70)
        print("\n|   AP   |  AP50  |  AP75  |")
        print("|:------:|:------:|:------:|")
        print(f"| {unseen_metrics['AP']:.3f} | {unseen_metrics['AP50']:.3f} | {unseen_metrics['AP75']:.3f} |")
        
        # LVIS-style Summary Table
        print("\n" + "="*70)
        print("SUMMARY (LVIS-style format):")
        print("="*70)
        print("\n| Category Split       |    AP |   AP50 |   AP75 |")
        print("|:---------------------|------:|-------:|-------:|")
        print(f"| All Classes ({num_all})   | {results['AP']:.2f} |  {results['AP50']:.2f} |  {results['AP75']:.2f} |")
        print(f"| Seen Classes ({num_seen})  | {seen_metrics['AP']:.2f} |  {seen_metrics['AP50']:.2f} |  {seen_metrics['AP75']:.2f} |")
        print(f"| Novel Classes ({num_unseen})  | {unseen_metrics['AP']:.2f} |  {unseen_metrics['AP50']:.2f} |  {unseen_metrics['AP75']:.2f} |")
        print("="*70)
        
        # ======================================================================
        # Copypaste Results Summary (matching LVIS format)
        # ======================================================================
        print("\n" + "="*60)
        print("Results Summary")
        print("="*60)
        print("copypaste: Task: {}".format(iou_type))
        print("copypaste: AP,AP50,AP75,APs,APm,APl")
        print("copypaste: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
            results['AP'], results['AP50'], results['AP75'], 
            results['APs'], results['APm'], results['APl']))
        print("copypaste: Seen_AP,Novel_AP,All_AP")
        print("copypaste: {:.4f},{:.4f},{:.4f}".format(
            seen_metrics['AP'], unseen_metrics['AP'], results['AP']))
        
        # Detailed LVIS-style per-class AP log
        self._logger.info("[Evaluator] Evaluation results for {}: ".format(iou_type))
        self._logger.info("|  AP   |  AP50  |  AP75  |  APs  |  APm  |  APl  |")
        self._logger.info("|:-----:|:------:|:------:|:-----:|:-----:|:-----:|")
        self._logger.info("| {:.3f} | {:.3f}  | {:.3f}  | {:.3f} | {:.3f} | {:.3f} |".format(
            results['AP'], results['AP50'], results['AP75'], 
            results['APs'], results['APm'], results['APl']))
        
        self._logger.info("[Evaluator] Evaluation results for {} (seen classes only): ".format(iou_type))
        self._logger.info("|  AP   |  AP50  |  AP75  |")
        self._logger.info("|:-----:|:------:|:------:|")
        self._logger.info("| {:.3f} | {:.3f}  | {:.3f}  |".format(
            seen_metrics['AP'], seen_metrics['AP50'], seen_metrics['AP75']))
        
        self._logger.info("[Evaluator] Evaluation results for {} (novel classes only): ".format(iou_type))
        self._logger.info("|  AP   |  AP50  |  AP75  |")
        self._logger.info("|:-----:|:------:|:------:|")
        self._logger.info("| {:.3f} | {:.3f}  | {:.3f}  |".format(
            unseen_metrics['AP'], unseen_metrics['AP50'], unseen_metrics['AP75']))
        
        return results
