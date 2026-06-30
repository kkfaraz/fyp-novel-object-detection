import logging
import argparse
import itertools
import copy
import torch
import os
import sys
import numpy as np

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(proj_path)

os.environ['DETECTRON2_DATASETS'] = os.environ.get('DETECTRON2_DATASETS', os.path.join(proj_path, 'datasets/DETECTRON2_DATASETS'))

from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
from collections import OrderedDict
from detectron2.utils.logger import log_every_n_seconds, create_small_table
from lvis import LVISEval
from datasets.register_lvis_val_subset import lvis_meta_val_subset


def tasks_from_predictions(predictions):
    for pred in predictions:
        if "segmentation" in pred:
            return ("bbox", "segm")
    return ("bbox",)

def _evaluate_predictions_on_lvis(
        logger, lvis_gt, lvis_results, iou_type, max_dets_per_image=None, class_names=None, known_class_ids=None
):
    """
    Same as the original implementation, except that extra evaluation on only known or only novel classes is performed
    if `known_class_ids` is provided. For that replaces object of `LVISEval` with `LVISEvalCustom`.
    """

    metrics = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl", "APr", "APc", "APf"],
    }[iou_type]

    if len(lvis_results) == 0:  # TODO: check if needed
        logger.warn("No predictions from the model!")
        return {metric: float("nan") for metric in metrics}

    if iou_type == "segm":
        lvis_results = copy.deepcopy(lvis_results)
        # When evaluating mask AP, if the results contain bbox, LVIS API will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in lvis_results:
            c.pop("bbox", None)

    if max_dets_per_image is None:
        max_dets_per_image = 500  # Default for LVIS dataset (increased from 300 for 1203 classes)

    from lvis import LVISEval, LVISResults

    logger.info(f"[Evaluator new] Evaluating with max detections per image = {max_dets_per_image}")
    lvis_results = LVISResults(lvis_gt, lvis_results, max_dets=max_dets_per_image)
    if known_class_ids is not None:
        lvis_eval = LVISEvalCustom(lvis_gt, lvis_results, iou_type, known_class_ids)
    else:
        lvis_eval = LVISEval(lvis_gt, lvis_results, iou_type)
    lvis_eval.run()
    lvis_eval.print_results()

    # Pull the standard metrics from the LVIS results
    results = lvis_eval.get_results()
    results = {metric: float(results[metric] * 100) for metric in metrics}
    logger.info("[Evaluator new] Evaluation results for {}: \n".format(iou_type) + create_small_table(results))

    if known_class_ids is not None:  # Print results for known and novel classes separately
        for results, subtitle in [
            (lvis_eval.results_known, "known classes only"),
            (lvis_eval.results_novel, "novel classes only"),
        ]:
            results = {metric: float(results[metric] * 100) for metric in metrics}
            logger.info("Evaluation results for {} ({}): \n".format(iou_type, subtitle) + create_small_table(results))

    return results

class LVISEvalCustom(LVISEval):
    """
    Extends `LVISEval` with printing results for known and novel classes only when `known_class_ids` is provided.
    """

    def __init__(self, lvis_gt, lvis_dt, iou_type="segm", known_class_ids=None):
        super().__init__(lvis_gt, lvis_dt, iou_type)

        # Remap categories list following the mapping applied to train data, - that is list all categories in a
        # consecutive order and use their indices; see: `lvis-api/lvis/eval.py` line 109:
        # https://github.com/lvis-dataset/lvis-api/blob/35f09cd7c5f313a9bf27b329ca80effe2b0c8a93/lvis/eval.py#L109
        if known_class_ids is None:
            self.known_class_ids = None
        else:
            self.known_class_ids = [self.params.cat_ids.index(c) for c in known_class_ids]

    def _summarize(
            self, summary_type, iou_thr=None, area_rng="all", freq_group_idx=None, subset_class_ids=None
    ):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if subset_class_ids is None:  # Use all classes
            subset_class_ids = list(range(len(self.params.cat_ids)))

        aidx = [
            idx
            for idx, _area_rng in enumerate(self.params.area_rng_lbl)
            if _area_rng == area_rng
        ]

        if summary_type == 'ap':
            s = self.eval["precision"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            if freq_group_idx is not None:
                subset_class_ids = list(set(subset_class_ids).intersection(self.freq_groups[freq_group_idx]))
                s = s[:, :, subset_class_ids, aidx]
            else:
                s = s[:, :, subset_class_ids, aidx]
        else:
            s = self.eval["recall"]
            if iou_thr is not None:
                tidx = np.where(iou_thr == self.params.iou_thrs)[0]
                s = s[tidx]
            s = s[:, subset_class_ids, aidx]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        return mean_s

    def summarize(self):
        """Extends the default version by supporting calculating the results only for the subset of classes."""

        if not self.eval:
            raise RuntimeError("Please run accumulate() first.")

        if self.known_class_ids is None:
            eval_groups = [(self.results, None)]
        else:
            cat_ids_mapped_list = list(range(len(self.params.cat_ids)))
            novel_class_ids = list(set(cat_ids_mapped_list).difference(self.known_class_ids))
            self.results_known = OrderedDict()
            self.results_novel = OrderedDict()
            eval_groups = [
                (self.results, None),
                (self.results_known, self.known_class_ids),
                (self.results_novel, novel_class_ids),
            ]

        max_dets = self.params.max_dets

        for container, subset_class_ids in eval_groups:
            container["AP"]   = self._summarize('ap', subset_class_ids=subset_class_ids)
            container["AP50"] = self._summarize('ap', iou_thr=0.50, subset_class_ids=subset_class_ids)
            container["AP75"] = self._summarize('ap', iou_thr=0.75, subset_class_ids=subset_class_ids)
            container["APs"]  = self._summarize('ap', area_rng="small", subset_class_ids=subset_class_ids)
            container["APm"]  = self._summarize('ap', area_rng="medium", subset_class_ids=subset_class_ids)
            container["APl"]  = self._summarize('ap', area_rng="large", subset_class_ids=subset_class_ids)
            container["APr"]  = self._summarize('ap', freq_group_idx=0, subset_class_ids=subset_class_ids)
            container["APc"]  = self._summarize('ap', freq_group_idx=1, subset_class_ids=subset_class_ids)
            container["APf"]  = self._summarize('ap', freq_group_idx=2, subset_class_ids=subset_class_ids)

            key = "AR@{}".format(max_dets)
            container[key] = self._summarize('ar', subset_class_ids=subset_class_ids)

            for area_rng in ["small", "medium", "large"]:
                key = "AR{}@{}".format(area_rng[0], max_dets)
                container[key] = self._summarize('ar', area_rng=area_rng, subset_class_ids=subset_class_ids)

        # ======================================================================
        # Detailed LVIS-Style Metrics Output (matching COCO format)
        # ======================================================================
        if self.known_class_ids is not None:
            r = self.results
            r_k = self.results_known
            r_n = self.results_novel
            total_cats = len(self.params.cat_ids)
            num_known = len(self.known_class_ids)
            num_novel = total_cats - num_known
            
            print("\n" + "="*70)
            print("LVIS OVD Evaluation Results - Novel Object Detection Pipeline")
            print("="*70 + "\n")
            
            # All classes - detailed metrics (COCO-style line-by-line)
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets} ] = {r['AP']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets={max_dets} ] = {r['AP50']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets={max_dets} ] = {r['AP75']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets={max_dets} ] = {r['APs']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets={max_dets} ] = {r['APm']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets={max_dets} ] = {r['APl']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets} catIds=  r] = {r['APr']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets} catIds=  c] = {r['APc']:.3f}")
            print(f" Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets} catIds=  f] = {r['APf']:.3f}")
            
            ar_key = f"AR@{max_dets}"
            ars_key = f"ARs@{max_dets}"
            arm_key = f"ARm@{max_dets}"
            arl_key = f"ARl@{max_dets}"
            
            print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets={max_dets} ] = {r[ar_key]:.3f}")
            print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets={max_dets} ] = {r[ars_key]:.3f}")
            print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets={max_dets} ] = {r[arm_key]:.3f}")
            print(f" Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets={max_dets} ] = {r[arl_key]:.3f}")
            
            # Summary tables
            print("\n" + "="*70)
            print(f"Evaluation results for bbox (ALL CLASSES - {total_cats} categories):")
            print("="*70)
            print("\n|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |  APr  |  APc   |  APf   |")
            print("|:------:|:------:|:------:|:-----:|:------:|:------:|:-----:|:------:|:------:|")
            print(f"| {r['AP']*100:.3f} | {r['AP50']*100:.3f} | {r['AP75']*100:.3f} | {r['APs']*100:.3f} | {r['APm']*100:.3f} | {r['APl']*100:.3f} | {r['APr']*100:.3f} | {r['APc']*100:.3f} | {r['APf']*100:.3f} |")
            
            print("\n" + "="*70)
            print(f"Evaluation results for bbox (KNOWN CLASSES - {num_known} categories):")
            print("="*70)
            print("\n|   AP   |  AP50  |  AP75  |  APr  |  APc   |  APf   |")
            print("|:------:|:------:|:------:|:-----:|:------:|:------:|")
            print(f"| {r_k['AP']*100:.3f} | {r_k['AP50']*100:.3f} | {r_k['AP75']*100:.3f} | {r_k['APr']*100:.3f} | {r_k['APc']*100:.3f} | {r_k['APf']*100:.3f} |")
            
            print("\n" + "="*70)
            print(f"Evaluation results for bbox (NOVEL CLASSES - {num_novel} categories):")
            print("="*70)
            print("\n|   AP   |  AP50  |  AP75  |  APr  |  APc   |  APf   |")
            print("|:------:|:------:|:------:|:-----:|:------:|:------:|")
            print(f"| {r_n['AP']*100:.3f} | {r_n['AP50']*100:.3f} | {r_n['AP75']*100:.3f} | {r_n['APr']*100:.3f} | {r_n['APc']*100:.3f} | {r_n['APf']*100:.3f} |")
            
            # Per-Frequency Results (LVIS-specific)
            print("\n" + "="*70)
            print("Per-Frequency Results (LVIS standard):")
            print("="*70)
            print("\n| Frequency Split |    AP |")
            print("|:----------------|------:|")
            print(f"| Frequent (f)    | {r['APf']*100:.2f} |")
            print(f"| Common (c)      | {r['APc']*100:.2f} |")
            print(f"| Rare (r)        | {r['APr']*100:.2f} |")
            
            # LVIS-style Summary Table
            print("\n" + "="*70)
            print("SUMMARY (LVIS-style format):")
            print("="*70)
            print(f"\n| Category Split              |    AP |   AP50 |   AP75 |")
            print(f"|:----------------------------|------:|-------:|-------:|")
            print(f"| All Classes ({total_cats})          | {r['AP']*100:.2f} |  {r['AP50']*100:.2f} |  {r['AP75']*100:.2f} |")
            print(f"| Known Classes ({num_known})          | {r_k['AP']*100:.2f} |  {r_k['AP50']*100:.2f} |  {r_k['AP75']*100:.2f} |")
            print(f"| Novel Classes ({num_novel})        | {r_n['AP']*100:.2f} |  {r_n['AP50']*100:.2f} |  {r_n['AP75']*100:.2f} |")
            print("="*70)
            
            # Copypaste Results Summary
            print("\n" + "="*60)
            print("Results Summary")
            print("="*60)
            print("copypaste: Task: bbox")
            print("copypaste: AP,AP50,AP75,APs,APm,APl,APr,APc,APf")
            print("copypaste: {:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(
                r['AP']*100, r['AP50']*100, r['AP75']*100, 
                r['APs']*100, r['APm']*100, r['APl']*100,
                r['APr']*100, r['APc']*100, r['APf']*100))
            print("copypaste: Known_AP,Novel_AP,All_AP")
            print("copypaste: {:.4f},{:.4f},{:.4f}".format(
                r_k['AP']*100, r_n['AP']*100, r['AP']*100))


def eval_predictions(predictions, lvis_data_split, known_class_ids):
    """
    Same as `LVISEvaluator`, code had to be re-copied to fix a reference to a new `_evaluate_predictions_on_lvis()`
    that is re-defined below.
    """
    from lvis import LVIS
    lvis_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    tasks = tasks_from_predictions(lvis_results)

    metadata = MetadataCatalog.get(lvis_data_split)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    json_file = PathManager.get_local_path(metadata.json_file)
    lvis_api = LVIS(json_file)

    results = OrderedDict()

    # LVIS evaluator can be used to evaluate results for COCO dataset categories.
    # In this case `_metadata` variable will have a field with COCO-specific category mapping.
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
        reverse_id_mapping = {
            v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
        }
        for result in lvis_results:
            result["category_id"] = reverse_id_mapping[result["category_id"]]
    else:
        # unmap the category ids for LVIS (from 0-indexed to 1-indexed)
        for result in lvis_results:
            result["category_id"] += 1

    logger.info("[Evaluator new] Evaluating predictions ...")
    for task in sorted(tasks):
        res = _evaluate_predictions_on_lvis(
            logger,
            lvis_api,
            lvis_results,
            task,
            max_dets_per_image=None,
            class_names=metadata.get("thing_classes"),
            known_class_ids=known_class_ids,
        )
        results[task] = res

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--lvis-data-split", type=str, default="lvis_v1_val")

    args = parser.parse_args()

    predictions = torch.load(args.predictions)
    print("Length of predictions: ", len(predictions))
    data_split = args.lvis_data_split

    known_class_ids=[3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 169, 173, 183,
                         207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                         611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                         964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                         1139, 1190, 1202]

    results = eval_predictions(predictions, data_split, known_class_ids)
