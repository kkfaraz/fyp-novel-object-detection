import os
import sys
import torch
import json
import argparse
from detectron2.data import DatasetCatalog, MetadataCatalog

# Add current dir to path to find imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Register datasets
import datasets.register_lvis_val_subset 
import datasets.register_coco_ovd

from evaluation import LVISEvaluatorCustom
from load_models import load_clip_model

# Known class IDs for LVIS OVD benchmark (80 base/known categories)
KNOWN_CLASS_IDS = [
    3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154,
    169, 173, 183, 207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445,
    469, 474, 496, 534, 569, 611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816,
    837, 881, 912, 923, 943, 961, 962, 964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079,
    1095, 1097, 1102, 1112, 1115, 1123, 1133, 1139, 1190, 1202
]

def evaluate_lvis(predictions_file, output_dir, data_split="lvis_v1_val"):
    print(f"Loading predictions from {predictions_file}...")
    try:
        predictions = torch.load(predictions_file, map_location="cpu")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return
        
    print(f"Loaded {len(predictions)} predictions.")
    
    # Create evaluator with known/novel class split
    evaluator = LVISEvaluatorCustom(
        dataset_name=data_split,
        distributed=False,
        output_dir=output_dir,
        max_dets_per_image=500,
        known_class_ids=KNOWN_CLASS_IDS
    )
    
    evaluator.reset()
    
    # Set the evaluator's internal state
    # LVISEvaluatorCustom's _eval_predictions expects list of dicts with "instances"
    print("Formatting predictions and running evaluation...")
    
    # Actually LVISEvaluatorCustom._eval_predictions expects the predictions 
    # to be a list of dicts. We have exactly this from pipeline_orchestrator.
    evaluator._eval_predictions(predictions)
    
    results = evaluator.evaluate()
    
    print("\n--- Detailed LVIS Evaluation Results ---")
    if results:
        for k, v in results.items():
            print(f"\n{k}:")
            if isinstance(v, dict):
                for metric, val in v.items():
                    print(f"  {metric}: {val}")
            else:
                print(f"  {v}")
                
        # Append to history file
        history_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Experiment_Results_History.txt"))
        try:
            with open(history_file, "a") as f:
                f.write("\n================================================================================\n")
                f.write(" NEW RUN EVALUATION RESULTS\n")
                f.write("================================================================================\n")
                f.write(f" Predictions file: {os.path.basename(predictions_file)}\n")
                for k, v in results.items():
                    f.write(f"\n{k}:\n")
                    if isinstance(v, dict):
                        for metric, val in v.items():
                            f.write(f"  {metric}: {val}\n")
        except Exception as e:
            print(f"Could not write to history file: {e}")
    else:
        print("Evaluation returned None.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LVIS predictions")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions .pkl or .pt file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for json results")
    parser.add_argument("--split", type=str, default="lvis_v1_val", help="Dataset split (e.g. lvis_v1_val)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    evaluate_lvis(args.predictions, args.output_dir, args.split)
