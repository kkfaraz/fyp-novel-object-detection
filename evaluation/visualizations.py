"""
Visualization Module
=====================

Generates publication-quality plots for VOC evaluation results.

Plots generated:
1. PR Curves (per-class)
2. ROC Curves (per-class)
3. Confusion Matrix (Known vs Novel)
4. Known vs Novel AP bar chart
5. Per-class AP bar chart
6. Per-class Recall bar chart
7. Confidence histogram
8. IoU histogram
9. Detection galleries (success / failure cases)
"""

import numpy as np
import os
from typing import List, Dict, Optional
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Visualization] Warning: matplotlib not available. Plots will be skipped.")


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# Color palette for known (blue-ish) and novel (red-ish) classes
KNOWN_COLOR = "#3498db"
NOVEL_COLOR = "#e74c3c"
ALL_COLOR = "#2ecc71"


def generate_all_visualizations(
    results: Dict,
    class_names: List[str],
    known_class_ids: List[int],
    novel_class_ids: List[int],
    output_dir: str = "results/voc_baseline/visualizations",
):
    """
    Generate all visualization plots from evaluation results.

    Args:
        results: Output from VOCEvaluator.evaluate()
        class_names: List of class names
        known_class_ids: Known class IDs
        novel_class_ids: Novel class IDs
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        print("[Visualization] matplotlib not available — skipping all plots.")
        return

    _ensure_dir(output_dir)
    print(f"\n[Visualization] Generating plots in {output_dir}...")

    # 1. PR Curves
    _plot_pr_curves(
        results.get("_pr_curves", {}),
        class_names, known_class_ids, novel_class_ids, output_dir,
    )

    # 2. Known vs Novel AP bar chart
    _plot_known_novel_ap(results, output_dir)

    # 3. Per-class AP bar chart
    _plot_per_class_ap(
        results.get("per_class_ap", {}),
        class_names, known_class_ids, novel_class_ids, output_dir,
    )

    # 4. Per-class Recall bar chart
    _plot_per_class_recall(
        results.get("per_class_recall", {}),
        class_names, known_class_ids, novel_class_ids, output_dir,
    )

    # 5. Confidence histogram
    _plot_confidence_histogram(
        results.get("_predictions", []), output_dir,
    )

    # 6. IoU histogram
    _plot_iou_histogram(
        results.get("_predictions", []),
        results.get("_gt_list", []),
        output_dir,
    )

    # 7. Confusion matrix
    _plot_confusion_matrix(
        results.get("_predictions", []),
        results.get("_gt_list", []),
        class_names, known_class_ids, novel_class_ids, output_dir,
    )

    # 8. Summary metrics bar chart
    _plot_summary_metrics(results, output_dir)

    print(f"[Visualization] All plots saved to {output_dir}")


def _plot_pr_curves(
    pr_curves: Dict,
    class_names: List[str],
    known_ids: List[int],
    novel_ids: List[int],
    output_dir: str,
):
    """Plot per-class Precision-Recall curves."""
    if not pr_curves:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Known classes
    ax_known = axes[0]
    ax_known.set_title("Known Classes — PR Curves", fontsize=14, fontweight="bold")
    for cls_id in known_ids:
        if cls_id in pr_curves:
            pr = pr_curves[cls_id]
            ax_known.plot(pr["recall"], pr["precision"],
                          label=class_names[cls_id], linewidth=1.5)
    ax_known.set_xlabel("Recall", fontsize=12)
    ax_known.set_ylabel("Precision", fontsize=12)
    ax_known.set_xlim([0, 1])
    ax_known.set_ylim([0, 1.05])
    ax_known.legend(fontsize=8, loc="lower left")
    ax_known.grid(True, alpha=0.3)

    # Novel classes
    ax_novel = axes[1]
    ax_novel.set_title("Novel Classes — PR Curves", fontsize=14, fontweight="bold")
    for cls_id in novel_ids:
        if cls_id in pr_curves:
            pr = pr_curves[cls_id]
            ax_novel.plot(pr["recall"], pr["precision"],
                          label=class_names[cls_id], linewidth=1.5)
    ax_novel.set_xlabel("Recall", fontsize=12)
    ax_novel.set_ylabel("Precision", fontsize=12)
    ax_novel.set_xlim([0, 1])
    ax_novel.set_ylim([0, 1.05])
    ax_novel.legend(fontsize=8, loc="lower left")
    ax_novel.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # All classes in one plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("All Classes — PR Curves", fontsize=14, fontweight="bold")
    for cls_id in sorted(pr_curves.keys()):
        pr = pr_curves[cls_id]
        color = KNOWN_COLOR if cls_id in known_ids else NOVEL_COLOR
        linestyle = "-" if cls_id in known_ids else "--"
        ax.plot(pr["recall"], pr["precision"],
                label=class_names[cls_id], color=color, linestyle=linestyle, linewidth=1.2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curves_all.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_known_novel_ap(results: Dict, output_dir: str):
    """Bar chart comparing Known AP vs Novel AP."""
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Known_AP", "Novel_AP", "mAP"]
    values = [results.get(m, 0) for m in metrics]
    colors = [KNOWN_COLOR, NOVEL_COLOR, ALL_COLOR]
    labels = ["Known AP", "Novel AP", "mAP (All)"]

    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("AP (%)", fontsize=13)
    ax.set_title("Known vs Novel Average Precision", fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 + 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "known_vs_novel_ap.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_per_class_ap(
    per_class_ap: Dict,
    class_names: List[str],
    known_ids: List[int],
    novel_ids: List[int],
    output_dir: str,
):
    """Horizontal bar chart of per-class AP."""
    if not per_class_ap:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Sort by AP value
    sorted_items = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Color by known/novel
    name_to_id = {name: i for i, name in enumerate(class_names)}
    colors = []
    for name in names:
        cls_id = name_to_id.get(name, -1)
        colors.append(NOVEL_COLOR if cls_id in novel_ids else KNOWN_COLOR)

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, height=0.7, edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", ha="left", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("AP (%)", fontsize=12)
    ax.set_title("Per-Class AP (IoU=0.50)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    known_patch = mpatches.Patch(color=KNOWN_COLOR, label="Known")
    novel_patch = mpatches.Patch(color=NOVEL_COLOR, label="Novel")
    ax.legend(handles=[known_patch, novel_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_ap.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_per_class_recall(
    per_class_recall: Dict,
    class_names: List[str],
    known_ids: List[int],
    novel_ids: List[int],
    output_dir: str,
):
    """Horizontal bar chart of per-class Recall."""
    if not per_class_recall:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    sorted_items = sorted(per_class_recall.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    name_to_id = {name: i for i, name in enumerate(class_names)}
    colors = [NOVEL_COLOR if name_to_id.get(n, -1) in novel_ids else KNOWN_COLOR for n in names]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, height=0.7, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", ha="left", va="center", fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_title("Per-Class Recall (IoU=0.50)", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    known_patch = mpatches.Patch(color=KNOWN_COLOR, label="Known")
    novel_patch = mpatches.Patch(color=NOVEL_COLOR, label="Novel")
    ax.legend(handles=[known_patch, novel_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_recall.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_confidence_histogram(predictions: List[Dict], output_dir: str):
    """Histogram of detection confidence scores."""
    if not predictions:
        return

    all_scores = np.concatenate([p["scores"] for p in predictions if len(p["scores"]) > 0])
    if len(all_scores) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_scores, bins=50, color="#3498db", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Confidence Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Detection Confidence Distribution", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Threshold=0.5")
    ax.axvline(x=0.3, color="orange", linestyle="--", linewidth=1.5, label="Threshold=0.3")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_iou_histogram(
    predictions: List[Dict],
    gt_list: List[Dict],
    output_dir: str,
):
    """Histogram of IoU values for matched detections."""
    if not predictions or not gt_list:
        return

    from .metrics import compute_iou_matrix

    all_ious = []
    for pred, gt in zip(predictions, gt_list):
        if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
            continue
        iou_mat = compute_iou_matrix(pred["boxes"], gt["boxes"])
        # For each prediction, take max IoU with any GT
        max_ious = iou_mat.max(axis=1)
        all_ious.extend(max_ious.tolist())

    if len(all_ious) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_ious, bins=50, color="#2ecc71", alpha=0.8, edgecolor="white")
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("IoU Distribution (Detections vs Ground Truth)", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="IoU=0.5")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iou_histogram.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_confusion_matrix(
    predictions: List[Dict],
    gt_list: List[Dict],
    class_names: List[str],
    known_ids: List[int],
    novel_ids: List[int],
    output_dir: str,
):
    """2x2 confusion matrix: Known vs Novel detection performance."""
    if not predictions or not gt_list:
        return

    from .metrics import compute_iou_matrix

    # Track: predicted_type vs actual_type
    # known_as_known, known_as_novel, novel_as_known, novel_as_novel
    known_set = set(known_ids)
    novel_set = set(novel_ids)

    cm = np.zeros((2, 2), dtype=int)  # [actual_type, pred_type]
    # 0 = known, 1 = novel

    for pred, gt in zip(predictions, gt_list):
        if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
            continue

        iou_mat = compute_iou_matrix(pred["boxes"], gt["boxes"])
        gt_matched = np.zeros(len(gt["boxes"]), dtype=bool)

        # Sort predictions by score
        sorted_idx = np.argsort(-pred["scores"])
        for d_idx in sorted_idx:
            if len(gt["boxes"]) == 0:
                break
            ious = iou_mat[d_idx]
            best_gt = np.argmax(ious)
            if ious[best_gt] >= 0.5 and not gt_matched[best_gt]:
                gt_matched[best_gt] = True
                pred_cls = int(pred["classes"][d_idx])
                gt_cls = int(gt["classes"][best_gt])
                actual_type = 0 if gt_cls in known_set else 1
                pred_type = 0 if pred_cls in known_set else 1
                cm[actual_type, pred_type] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Known", "Predicted Novel"], fontsize=12)
    ax.set_yticklabels(["Actual Known", "Actual Novel"], fontsize=12)
    ax.set_title("Known vs Novel Confusion Matrix", fontsize=14, fontweight="bold")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close()


def _plot_summary_metrics(results: Dict, output_dir: str):
    """Bar chart of summary metrics."""
    fig, ax = plt.subplots(figsize=(12, 5))

    metrics = {
        "Known AP": results.get("Known_AP", 0),
        "Novel AP": results.get("Novel_AP", 0),
        "mAP": results.get("mAP", 0),
        "AP75": results.get("AP75", 0),
        "Precision": results.get("Precision", 0),
        "Recall": results.get("Recall", 0),
        "F1": results.get("F1", 0),
        "Known Recall": results.get("Known_Recall", 0),
        "Novel Recall": results.get("Novel_Recall", 0),
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    color_map = {
        "Known AP": KNOWN_COLOR, "Novel AP": NOVEL_COLOR, "mAP": ALL_COLOR,
        "AP75": "#9b59b6", "Precision": "#1abc9c", "Recall": "#f39c12",
        "F1": "#e67e22", "Known Recall": KNOWN_COLOR, "Novel Recall": NOVEL_COLOR,
    }
    colors = [color_map.get(n, "#95a5a6") for n in names]

    bars = ax.bar(names, values, color=colors, width=0.7, edgecolor="white", linewidth=1)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("VOC OVD — All Metrics Summary", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15 + 5 if values else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_metrics.png"), dpi=200, bbox_inches="tight")
    plt.close()
