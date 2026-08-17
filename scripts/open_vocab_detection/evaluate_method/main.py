"""
FYP: COCO Open-Vocabulary Detection - VLRM + OT Fusion Version
================================================================
Updated to use Shared PipelineOrchestrator for Checkpointing/Resume.
"""
import os
import sys
import warnings
import json
import signal
import gc
import argparse
import torch
from pathlib import Path

proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(proj_path)

# IMPORTANT: Set DETECTRON2_DATASETS *before* importing register_coco_ovd_dataset,
# because that module reads the env var at import time.
_params_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")
with open(_params_path, "r") as _f:
    _json_params = json.load(_f)
os.environ['DETECTRON2_DATASETS'] = os.path.join(proj_path, _json_params["detectron2_dir"])

from groundingdino.util.inference import load_model
from load_models import load_sam_model, load_maskrcnn_model
from utils import get_ovd_id_to_coco_id

# Import pipelines
from scripts.pipeline_orchestrator import PipelineOrchestrator
from coco_pipeline_stages import run_stage1_coco, run_stage2_coco, run_stage3_coco
from datasets.register_coco_ovd_dataset import coco_meta
from scripts.open_vocab_detection.coco_eval_utils.custom_coco_eval import CustomCOCOEvaluator
from scripts.open_vocab_detection.coco_eval_utils.coco_ovd_split import categories_seen, categories_unseen
from scripts.gpu_utils import get_device, log_device_info, validate_all_models, log_gpu_memory, clear_gpu_memory

from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts, DatasetMapper
from detectron2.evaluation import print_csv_format
import detectron2.data.transforms as T
from segment_anything.utils.transforms import ResizeLongestSide

def signal_handler(sig, frame):
    print("\n\n[INFO] Interrupted. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def get_gpu_mem():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0

# COCO Classes
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

def main():
    parser = argparse.ArgumentParser(description='COCO OVD Evaluation')
    parser.add_argument('--config', type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--weights', type=str, default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--coco-path', type=str, required=True, help='Path to COCO dataset root')
    parser.add_argument('--output-dir', type=str, default="outputs_coco", help='Directory to save outputs')
    parser.add_argument('--exp-name', type=str, default="default", help='Experiment name')
    
    parser.add_argument('--chunk', type=int, default=None, help='Which chunk to process (1-N). If not specified, runs all.')
    parser.add_argument('--all-chunks', action='store_true', help='Process all chunks sequentially')
    parser.add_argument('--num-chunks', type=int, default=5, help='Total number of chunks')
    
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoints (Default: True)')
    parser.add_argument('--force_recompute', action='store_true', help='Force re-run stages')
    parser.add_argument('--start_stage', type=int, default=0, choices=[0, 1, 2, 3], help='0=Auto/Resume')
    
    parser.add_argument('--split', type=str, default='coco_ovd_val', help='Detectron2 registered dataset name (e.g. coco_ovd_val, coco_2017_val)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    # Debugging and Logging
    parser.add_argument('--debug_mode', action='store_true', default=True, help='Enable detailed module tracking')
    parser.add_argument('--verbose_level', type=str, default="medium", choices=["low", "medium", "high"], help='Logging detail level')
    parser.add_argument("--vlrm-cache-dir", type=str, default="cache/vlrm_outputs_coco", help="Directory to store/load VLRM captions.")
    
    args = parser.parse_args()
    
    # Setup Env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(script_dir, "params.json")
    
    with open(params_path, "r") as f:
        json_params = json.load(f)

    detectron2_dir = os.path.join(proj_path, json_params["detectron2_dir"])
    cfg_file = os.path.join(proj_path, json_params["cfg_file"])
    sam_checkpoint = os.path.join(proj_path, json_params["sam_checkpoint"])
    gdino_checkpoint = os.path.join(proj_path, json_params["gdino_checkpoint"]) 
    
    os.environ['DETECTRON2_DATASETS'] = detectron2_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    output_abs_path = os.path.abspath(os.path.join(script_dir, "../../../", args.output_dir))
    Path(output_abs_path).mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    log_device_info()
    
    print(f"\n{'='*60}")
    print("FYP: COCO OVD - Memory-Safe 3-Stage Pipeline (Resumable)")
    print(f"{'='*60}")
    if torch.cuda.is_available():
        print(f"[Memory] Initial GPU: {get_gpu_mem():.2f} GB")
        
    # ==========================================================================
    # Load Models
    # ==========================================================================
    # 1. GDINO
    print("\n[1/3] GroundingDINO (GPU)...")
    clear_gpu_memory()
    
    if "GroundingDINO_SwinT_OGC.py" in args.config:
        gdino_config_path = os.path.join(proj_path, "cfg/GroundingDINO/GroundingDINO_SwinT_OGC.py")
    else:
        gdino_config_path = args.config
        
    model = load_model(gdino_config_path, gdino_checkpoint)
    model = model.to(device)
    model.eval()
    
    # 2. Mask-RCNN (GPU — was CPU, now enforced GPU for full pipeline GPU execution)
    print("[2/3] Mask-RCNN (GPU)...")
    rcnn_weight_path = os.path.join(proj_path, "MaskRCNN_COCO_OVD")
    maskrcnn_model, _ = load_maskrcnn_model(cfg_file, rcnn_weight_path, device)
    
    # 3. SAM
    print("[3/3] SAM (GPU)...")
    sam = load_sam_model(device, sam_checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    clear_gpu_memory()
    
    # GPU Validation: Verify all models
    validate_all_models({
        "GroundingDINO": model,
        "Mask-RCNN": maskrcnn_model,
        "SAM": sam,
    }, device)
    log_gpu_memory("after_model_loading")
    
    # ==========================================================================
    # Data Loader
    # ==========================================================================
    test_loader = build_detection_test_loader(
        dataset=get_detection_dataset_dicts(names=args.split, filter_empty=False),
        mapper=DatasetMapper(
            is_train=False,
            augmentations=[T.ResizeShortestEdge(short_edge_length=600, max_size=1000)],
            image_format="BGR",
        ),
        num_workers=0,
    )
    all_data = list(test_loader)
    total_images = len(all_data)
    
    # ==========================================================================
    # Evaluator & Params
    # ==========================================================================
    coco_evaluator = CustomCOCOEvaluator(dataset_name=args.split)
    ovd_id_to_coco_id = get_ovd_id_to_coco_id()
    
    param_dict = {
        "visualize": json_params["visualize"],
        "out_dir": output_abs_path,
        "data_split": args.split,
        "device": device,
        "ovd_id_to_coco_id": ovd_id_to_coco_id,
        "coco_classes": COCO_CLASSES,
        "maskrcnn_model": maskrcnn_model,
        "sam": sam,
        "resize_transform": resize_transform,
        "verbose": args.verbose,
        "debug_mode": args.debug_mode,
        "verbose_level": args.verbose_level,
        "vlrm_cache_dir": args.vlrm_cache_dir,
        "stage2_threshold": json_params.get("stage2_threshold", 0.20),
        "use_srm": json_params.get("use_srm", True)
    }
    
    # Text Prompt
    seen_names = [x['name'] for x in categories_seen]
    unseen_names = [x['name'] for x in categories_unseen]
    ovd_classes = seen_names + unseen_names
    text_prompt = " . ".join([c.lower().replace("_", " ") for c in ovd_classes])
    
    # ==========================================================================
    # Chunking Logic
    # ==========================================================================
    chunks_to_run = []
    if args.chunk is not None:
         chunks_to_run = [args.chunk]
    elif args.all_chunks:
         chunks_to_run = list(range(1, args.num_chunks + 1))
    else:
         chunks_to_run = [0] # Single chunk
         
    num_chunks = args.num_chunks
    chunk_size = (total_images + num_chunks - 1) // num_chunks
    
    for chunk_id in chunks_to_run:
        if chunk_id == 0:
            chunk_data = all_data
            current_exp_name = args.exp_name
            print(f"\nProcessing Entire Dataset ({len(chunk_data)} images)")
        else:
            start_idx = (chunk_id - 1) * chunk_size
            end_idx = min(chunk_id * chunk_size, total_images)
            chunk_data = all_data[start_idx:end_idx]
            current_exp_name = f"{args.exp_name}_chunk{chunk_id}"
            print(f"\nProcessing Chunk {chunk_id}/{num_chunks} ({len(chunk_data)} images)")
            
        print(f"Experiment Name: {current_exp_name}")
        
        # Initialize Orchestrator
        orchestrator = PipelineOrchestrator(
             output_dir=args.output_dir,
             exp_name=current_exp_name,
             device=device,
             force_recompute=args.force_recompute
        )
        
        # Save Metadata
        orchestrator.save_metadata(vars(args))
        
        # Stage 1
        s1_results = orchestrator.run_stage(
             stage_id=1,
             description="GDINO + MaskRCNN",
             worker_func=run_stage1_coco,
             dataloader=chunk_data,
             model=model,
             text_prompt=text_prompt,
             param_dict=param_dict
        )
        
        # Stage 2
        s2_results = orchestrator.run_stage(
             stage_id=2,
             description="VLRM + CLIP",
             worker_func=run_stage2_coco,
             stage1_results=s1_results,
             param_dict=param_dict
        )
        
        # Stage 3
        s3_metrics = orchestrator.run_stage(
             stage_id=3,
             description="Optimization & Evaluation",
             worker_func=run_stage3_coco,
             stage1_results=s1_results,
             stage2_results=s2_results,
             param_dict=param_dict,
             evaluator=coco_evaluator
        )
        
        if s3_metrics:
            print_detailed_metrics(s3_metrics)

def print_detailed_metrics(metrics_dict):
    if not metrics_dict or "bbox" not in metrics_dict:
        print("\nFinal Metrics:")
        print(metrics_dict)
        return
        
    res = metrics_dict["bbox"]
    
    print("\n" + "="*70)
    print("COCO OVD Evaluation Results - Novel Object Detection Pipeline")
    print("="*70)
    
    print("\n" + "="*70)
    print("Evaluation results for bbox (ALL CLASSES - 65 categories):")
    print("="*70)
    print("\n|   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |")
    print("|:------:|:------:|:------:|:-----:|:------:|:------:|")
    print(f"| {res.get('AP', float('nan')):.3f} | {res.get('AP50', float('nan')):.3f} | {res.get('AP75', float('nan')):.3f} | {res.get('APs', float('nan')):.3f} | {res.get('APm', float('nan')):.3f} | {res.get('APl', float('nan')):.3f} |")
    
    print("\n" + "="*70)
    print("Evaluation results for bbox (SEEN CLASSES - 48 categories):")
    print("="*70)
    print("\n|   AP   |  AP50  |  AP75  |")
    print("|:------:|:------:|:------:|")
    print(f"| {res.get('AP-seen', float('nan')):.3f} | {res.get('AP50-seen', float('nan')):.3f} | {res.get('AP75-seen', float('nan')):.3f} |")
    
    print("\n" + "="*70)
    print("Evaluation results for bbox (UNSEEN/NOVEL CLASSES - 17 categories):")
    print("="*70)
    print("\n|   AP   |  AP50  |  AP75  |")
    print("|:------:|:------:|:------:|")
    print(f"| {res.get('AP-unseen', float('nan')):.3f} | {res.get('AP50-unseen', float('nan')):.3f} | {res.get('AP75-unseen', float('nan')):.3f} |")
    
    print("\n" + "="*70)
    print("SUMMARY (LVIS-style format):")
    print("="*70)
    print("\n| Category Split       |    AP |   AP50 |   AP75 |")
    print("|:---------------------|------:|-------:|-------:|")
    print(f"| All Classes (65)     | {res.get('AP', float('nan')):.2f} |  {res.get('AP50', float('nan')):.2f} |  {res.get('AP75', float('nan')):.2f} |")
    print(f"| Seen Classes (48)    | {res.get('AP-seen', float('nan')):.2f} |  {res.get('AP50-seen', float('nan')):.2f} |  {res.get('AP75-seen', float('nan')):.2f} |")
    print(f"| Novel Classes (17)   | {res.get('AP-unseen', float('nan')):.2f} |  {res.get('AP50-unseen', float('nan')):.2f} |  {res.get('AP75-unseen', float('nan')):.2f} |")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
