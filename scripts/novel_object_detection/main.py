
import argparse
import os
import sys
import pickle
import torch
import json
import warnings
from pathlib import Path

# Add project root and parent scripts directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_proj_path = os.path.abspath(os.path.join(_script_dir, "../../"))
sys.path.insert(0, _proj_path)
sys.path.insert(0, os.path.join(_script_dir, '..'))
# Also add the inner detectron2 package dir (local source checkout)
_d2_inner = os.path.join(_proj_path, "detectron2", "detectron2")
if os.path.isdir(_d2_inner):
    sys.path.insert(0, os.path.join(_proj_path, "detectron2"))

from gpu_utils import get_device, log_device_info, validate_all_models, log_gpu_memory, clear_gpu_memory

# ── Set DETECTRON2_DATASETS BEFORE importing detectron2 ──────────────────────
# Detectron2 resolves dataset paths at import time from this env var.
# It MUST be set before any detectron2.data imports.
_params_path = os.path.join(_script_dir, "params.json")

with open(_params_path, 'r') as _f:
    _params_boot = json.load(_f)

_detectron2_dir = os.path.join(_proj_path, _params_boot.get("detectron2_dir", "./datasets/DETECTRON2_DATASETS"))
os.environ['DETECTRON2_DATASETS'] = _detectron2_dir
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
torch.backends.cudnn.benchmark = True

# Validate LVIS dataset path exists (skip for VOC)
_lvis_split = _params_boot.get("data_split", _params_boot.get("lvis_data_split", "lvis_v1_val"))
_is_voc = False
for arg in sys.argv:
    if "--split" in arg:
        idx = sys.argv.index(arg)
        if len(sys.argv) > idx + 1:
            if "voc" in sys.argv[idx + 1].lower():
                _is_voc = True
if not _is_voc and "voc" in _lvis_split.lower():
    _is_voc = True

if _is_voc:
    print(f"[Config] VOC mode detected. Skipping LVIS dataset path validation.")
else:
    _lvis_json = os.path.join(_detectron2_dir, "lvis", f"{_lvis_split}.json")
    if not os.path.isfile(_lvis_json):
        print(f"ERROR: LVIS annotation file not found at: {_lvis_json}")
        print(f"  DETECTRON2_DATASETS is set to: {_detectron2_dir}")
        print(f"  Expected structure: {_detectron2_dir}/lvis/{_lvis_split}.json")
        print(f"  Available files in lvis dir:")
        _lvis_dir = os.path.join(_detectron2_dir, "lvis")
        if os.path.isdir(_lvis_dir):
            for f in os.listdir(_lvis_dir):
                print(f"    - {f}")
        else:
            print(f"    Directory does not exist: {_lvis_dir}")
        sys.exit(1)
    print(f"[Config] DETECTRON2_DATASETS = {_detectron2_dir}")
    print(f"[Config] LVIS split = {_lvis_split}, annotation = {_lvis_json}")

if _is_voc:
    try:
        from detectron2.data.datasets.pascal_voc import register_pascal_voc
        voc_data_root = os.path.join(_proj_path, "datasets/DETECTRON2_DATASETS/VOCdevkit")
        if not os.path.isdir(voc_data_root):
            voc_data_root = os.path.join(_proj_path, "datasets/VOCdevkit")
        register_pascal_voc("voc_custom_2007_test", os.path.join(voc_data_root, "VOC2007"), "test", "2007")
        print("[Dataset] Registered custom VOC datasets successfully")
    except Exception as e:
        print(f"[Dataset] Warning: Could not register VOC datasets: {e}")
else:
    try:
        sys.path.append(_proj_path)
        import datasets.register_lvis_val_subset
        print("[Dataset] Registered lvis_v1_val_subset successfully")
    except Exception as e:
        print(f"[Dataset] Warning: Could not register lvis_v1_val_subset: {e}")


from groundingdino.util.inference import load_model
from load_models import load_sam_model, load_fully_supervised_trained_model, load_torchvision_maskrcnn, load_clip_model
# from utils import get_dataset_dictionaries # Use detectron2 directly if missing
from pipeline_orchestrator import PipelineOrchestrator
from lvis_pipeline_stages import run_stage1_lvis, run_stage2_lvis, run_stage3_lvis

from detectron2.data import get_detection_dataset_dicts
from scripts.novel_object_detection.evaluation import LVISEvaluatorCustom

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="LVIS Novel Object Detection Pipeline (3-Stage Resumable)")
    parser.add_argument('--config', type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument('--weights', type=str, default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument('--coco-path', type=str, default='', help='Path to COCO dataset root (unused, uses DETECTRON2_DATASETS)')
    parser.add_argument('--output-dir', type=str, default="outputs_lvis", help='Directory to save outputs')
    parser.add_argument('--exp-name', type=str, default="test_exp", help='Experiment name')
    
    parser.add_argument('--chunk', type=int, default=None, help='Which chunk to process (1-N). If not specified, runs all.')
    parser.add_argument('--all-chunks', action='store_true', help='Process all chunks sequentially')
    parser.add_argument('--num-chunks', type=int, default=5, help='Total number of chunks')
    
    parser.add_argument('--resume', action='store_true', default=True, help='Resume from checkpoints (Default: True)')
    parser.add_argument('--force_recompute', action='store_true', help='Force re-run stages')
    parser.add_argument('--start_stage', type=int, default=0, choices=[0, 1, 2, 3], help='0=Auto/Resume')
    
    parser.add_argument('--split', type=str, default='lvis_v1_val', help='Detectron2 registered dataset name (e.g. coco_ovd_val, coco_2017_val)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logs')
    
    # Debugging and Logging
    parser.add_argument('--debug_mode', action='store_true', default=True, help='Enable detailed module tracking')
    parser.add_argument('--verbose_level', type=str, default="medium", choices=["low", "medium", "high"], help='Logging detail level')
    parser.add_argument("--vlrm-cache-dir", type=str, default="cache/vlrm_outputs_lvis", help="Directory to store/load VLRM captions.")
    
    args = parser.parse_args()
    
    # Setup Paths (already loaded at top-level for env var)
    script_dir = _script_dir
    params_path = _params_path
    
    with open(params_path, 'r') as f:
        params = json.load(f)
        
    proj_path = _proj_path
    
    # Output Dir
    # params["output_dir"] might exist but arg overrides?
    # Use arg.
    full_output_dir = os.path.join(proj_path, args.output_dir)
    Path(full_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine Chunk
    # Use standard Detectron2 loader logic or params['data_split']
    data_split = args.split if args.split else params.get("data_split", params.get("lvis_data_split", "lvis_v1_val"))
    dataset_dicts = get_detection_dataset_dicts(names=data_split, filter_empty=False)
    
    # Load LVIS classes (Need access to LVIS API or metadata)
    from detectron2.data import MetadataCatalog
    meta = MetadataCatalog.get(data_split)
    lvis_classes = meta.thing_classes
    
    # Chunking
    total_images = len(dataset_dicts)
    if args.chunk is not None:
        chunk_size = (total_images + args.num_chunks - 1) // args.num_chunks
        start_idx = (args.chunk - 1) * chunk_size
        end_idx = min(args.chunk * chunk_size, total_images)
        chunk_dicts = dataset_dicts[start_idx:end_idx]
        current_exp_name = f"{args.exp_name}_chunk{args.chunk}"
        print(f"Processing Chunk {args.chunk}/{args.num_chunks}: {len(chunk_dicts)} images")
    else:
        chunk_dicts = dataset_dicts
        current_exp_name = args.exp_name
        print(f"Processing All {len(dataset_dicts)} images")
        
    device = get_device()
    log_device_info()
    
    # Initialize Orchestrator
    orchestrator = PipelineOrchestrator(
        output_dir=full_output_dir,
        exp_name=current_exp_name,
        device=device,
        force_recompute=args.force_recompute
    )
    orchestrator.save_metadata(vars(args))
    
    print("Loading Models...")
    clear_gpu_memory()
    # GDINO
    gdino_config_path = os.path.join(proj_path, args.config.replace("GroundingDINO/", "cfg/GroundingDINO/").replace("config/", ""))
    # Fallback if specific:
    if "GroundingDINO_SwinT_OGC.py" in args.config:
        gdino_config_path = os.path.join(proj_path, "cfg/GroundingDINO/GroundingDINO_SwinT_OGC.py")
    elif not os.path.exists(gdino_config_path):
        gdino_config_path = os.path.join(proj_path, "cfg/GroundingDINO/GDINO.py")
        
    gdino_model = load_model(gdino_config_path, params["gdino_checkpoint"])
    gdino_model = gdino_model.to(device)
    
    # Use LVIS config (clean) with COCO OVD weights for better novel detection
    cfg_file_path = os.path.join(proj_path, params["cfg_file"])
    rcnn_weight_path = os.path.join(proj_path, params["rcnn_weight_dir"])
    print(f"[Model] Config: {cfg_file_path}")
    print(f"[Model] Weights: {rcnn_weight_path}")
    maskrcnn_model, cfg = load_fully_supervised_trained_model(cfg_file_path, rcnn_weight_path)
    sam_model = load_sam_model(device, params["sam_checkpoint"])
    
    # Load TorchVision Mask-RCNN for RPN background proposals (critical for Stage 2)
    torchvision_maskrcnn = load_torchvision_maskrcnn(device)
    
    # Build GDINO text prompts and positive maps for LVIS categories
    from utils import get_text_prompt_list_for_g_dino, get_coco_to_lvis_mapping
    from segment_anything.utils.transforms import ResizeLongestSide
    
    class_len_per_prompt = params.get("class_len_per_prompt", 81)
    lvis_data_split = data_split
    
    tokenizer = gdino_model.tokenizer
    text_prompt_list, positive_map_list = get_text_prompt_list_for_g_dino(lvis_data_split, tokenizer, class_len_per_prompt)
    coco_to_lvis = get_coco_to_lvis_mapping(cfg, lvis_data_split)
    resize_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    
    # GPU Validation: Verify all models are on correct device
    validate_all_models({
        "GroundingDINO": gdino_model,
        "Mask-RCNN (Detectron2)": maskrcnn_model,
        "Mask-RCNN (TorchVision)": torchvision_maskrcnn,
        "SAM": sam_model,
    }, device)
    log_gpu_memory("after_model_loading")
    
    is_voc = "voc" in data_split.lower()

    param_dict = {
        "max_dets_per_image": 100 if is_voc else 500,
        "max_before_sam": 300 if is_voc else 500,
        "model": gdino_model,
        "gdino_model": gdino_model,
        "maskrcnn_model": maskrcnn_model,
        "rcnn_model": maskrcnn_model,
        "torchvision_maskrcnn": torchvision_maskrcnn,
        "sam": sam_model,
        "resize_transform": resize_transform,
        "box_threshold": params.get("box_threshold", 0.35),
        "text_threshold": params.get("text_threshold", 0.25),
        "nms_threshold": params.get("nms_threshold", 0.5),
        "caption": None,
        "device": device,
        "lvis_classes": lvis_classes,
        "verbose": args.verbose,
        "debug_mode": args.debug_mode,
        "verbose_level": args.verbose_level,
        "vlrm_cache_dir": args.vlrm_cache_dir,
        "force_recompute": args.force_recompute,
        "stage2_threshold": params.get("stage2_threshold", 0.12),
        "use_srm": params.get("use_srm", True),
        "use_apf": params.get("use_apf", True),
        "apf_source_weights": params.get("apf_source_weights", {0: 0.55, 1: 0.50, 2: 0.40}),
        "apf_iou_threshold": params.get("apf_iou_threshold", 0.5),
        "apf_score_threshold": params.get("apf_score_threshold", 0.05),
        "apf_max_boxes": params.get("apf_max_boxes", 500),
        "use_saeg": params.get("use_saeg", False),
        "recompute_stage2_scores": params.get("recompute_stage2_scores", True),
        # LVIS-specific keys required by ground_dino_utils.py
        "coco_to_lvis": coco_to_lvis,
        "text_prompt_list": text_prompt_list,
        "positive_map_list": positive_map_list,
        "class_len_per_prompt": class_len_per_prompt,
        "lvis_data_split": lvis_data_split,
        "out_dir": full_output_dir,
        "visualize": params.get("visualize", False),
    }
    
    # Run Stage 1
    param_dict_s1 = param_dict.copy()
    param_dict_s1["sam"] = None

    s1_results = orchestrator.run_stage(
        stage_id=1,
        description="GDINO + MaskRCNN",
        worker_func=run_stage1_lvis,
        dataloader=chunk_dicts,
        model=gdino_model,
        text_prompt=None, # text_prompt_list is passed via param_dict in LVIS
        param_dict=param_dict_s1
    )
    
    # ── VRAM Reclaim after Stage 1 ─────────────────────────────────────────────
    # Stage 2 needs VLRM (BLIP-2 ~5GB FP16) + SigLIP (~1.5GB FP16).
    # Offload all Stage 1 models to CPU to reclaim ~5GB of VRAM for Stage 2 models.
    import gc as _gc
    if torchvision_maskrcnn is not None:
        torchvision_maskrcnn.cpu()
    if gdino_model is not None:
        gdino_model.cpu()
    if maskrcnn_model is not None:
        maskrcnn_model.cpu()
    if sam_model is not None:
        sam_model.cpu()
        
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[VRAM] All Stage 1 models (GDINO, Mask-RCNNs, SAM) offloaded to CPU after Stage 1 — VRAM freed for Stage 2 models")
    log_gpu_memory("after_stage1_offload")
    
    # Run Stage 2 (with checkpoint resume)
    stage2_checkpoint = os.path.join(full_output_dir, f"stage2_checkpoint_{current_exp_name}.pkl")
    if not args.force_recompute and args.resume and os.path.exists(stage2_checkpoint):
        print(f"[Checkpoint] Loading Stage 2 results from {stage2_checkpoint}")
        with open(stage2_checkpoint, "rb") as f:
            s2_results = pickle.load(f)
        print(f"[Checkpoint] Loaded {len(s2_results)} Stage 2 results. Skipping Stage 2.")
    else:
        s2_results = orchestrator.run_stage(
            stage_id=2,
            description="VLRM + CLIP",
            worker_func=run_stage2_lvis,
            stage1_results=s1_results,
            param_dict=param_dict
        )
        print(f"[Checkpoint] Saving Stage 2 results to {stage2_checkpoint}")
        with open(stage2_checkpoint, "wb") as f:
            pickle.dump(s2_results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Checkpoint] Saved {len(s2_results)} Stage 2 results.")
    
    # Initialize Evaluator before Stage 3
    if "voc" in data_split.lower():
        from datasets.voc.voc_class_utils import get_known_novel_split, VOC_CLASSES
        # Use split1 for known/novel split on VOC
        known_class_ids_voc, novel_class_ids_voc = get_known_novel_split("split1")
        
        gt_annotations = []
        for dict_obj in chunk_dicts:
            gt_annotations.append({
                "image_id": dict_obj["image_id"],
                "annotations": dict_obj.get("annotations", [])
            })
            
        from evaluation.voc_evaluation import VOCEvaluator
        evaluator = VOCEvaluator(
            class_names=VOC_CLASSES,
            known_class_ids=known_class_ids_voc,
            novel_class_ids=novel_class_ids_voc,
            output_dir=os.path.join(full_output_dir, "experiments", current_exp_name),
            gt_annotations=gt_annotations,
            use_07_metric=("2007" in data_split)
        )
    else:
        # Compute known_class_ids DYNAMICALLY from coco_to_lvis mapping to guarantee correctness
        known_class_ids = sorted(set(v for v in coco_to_lvis.values() if v >= 0))
        print(f"Initializing Evaluator with {len(known_class_ids)} known / {1203-len(known_class_ids)} novel class IDs...")
        evaluator = LVISEvaluatorCustom(
            dataset_name=data_split,
            distributed=False,
            output_dir=os.path.join(full_output_dir, "experiments", current_exp_name),
            max_dets_per_image=500,
            known_class_ids=known_class_ids
        )
    evaluator.reset()
    
    # Clear GPU memory before Stage 3 — SAM + SRM on full resolution is heavy
    clear_gpu_memory()
    print(f"[Memory] Cleared cache before Stage 3")
    
    # Move SAM model back to GPU for Stage 3
    if sam_model is not None:
        print("[VRAM] Moving SAM back to GPU for Stage 3")
        sam_model = sam_model.to(device)
        param_dict["sam"] = sam_model
        
    # Run Stage 3
    s3_results = orchestrator.run_stage(
        stage_id=3,
        description="Optimization (OT/SAM/SRM)",
        worker_func=run_stage3_lvis,
        stage1_results=s1_results,
        stage2_results=s2_results,
        param_dict=param_dict,
        evaluator=evaluator
    )

    print("\n" + "="*70)
    print("Evaluation Complete")
    print("="*70)

    if "voc" in data_split.lower():
        # Save results to JSON (excluding private metadata keys)
        clean_results = {k: v for k, v in s3_results.items() if not k.startswith("_")}
        results_path = os.path.join(full_output_dir, "evaluation_metrics.json")
        with open(results_path, "w") as f:
            json.dump(clean_results, f, indent=4)
        print(f"Results successfully saved to {results_path}")

        # Generate visualizations
        from datasets.voc.voc_class_utils import get_known_novel_split, VOC_CLASSES
        known_class_ids_voc, novel_class_ids_voc = get_known_novel_split("split1")
        print("\nGenerating evaluation visualizations...")
        from evaluation.visualizations import generate_all_visualizations
        generate_all_visualizations(
            results=s3_results,
            class_names=VOC_CLASSES,
            known_class_ids=known_class_ids_voc,
            novel_class_ids=novel_class_ids_voc,
            output_dir=os.path.join(full_output_dir, "visualizations")
        )
        print("Visualizations saved.")

if __name__ == "__main__":
    main()
