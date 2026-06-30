"""
Backend API Server for Cooperative Novel Object Detection
=========================================================

FastAPI server wrapping the existing 3-stage pipeline:
  Stage 1: GroundingDINO + Mask R-CNN → Known detections + Background ROIs
  Stage 2: VLRM + CLIP → Novel object labelling
  Stage 3: Hybrid OT + SAM + SRM → Refined detections

Usage:
    cd frontend/backend
    python server.py

The server will start on port 8000. Models are loaded lazily on first request.
"""

import os
import sys
import time
import json
import base64
import logging
import traceback
import tempfile
from io import BytesIO
from pathlib import Path
import gc
from typing import List, Dict, Any, Optional, Union, Set

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
NOD_DIR = SCRIPTS_DIR / "novel_object_detection"

# Force the working directory to the project root
# This fixes "file not found" errors for hardcoded relative paths in the pipeline (e.g. cfg/GroundingDINO/GDINO.py)
os.chdir(str(PROJECT_ROOT))

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(NOD_DIR))

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nod-server")

# ─── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Cooperative NOD API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pipeline State ──────────────────────────────────────────────────────────
STATE: Dict[str, Any] = {
    "initialized": False,
    "loading": False,
    "error": None,
    "model": None,              # GroundingDINO model
    "text_prompt_list": None,
    "param_dict": {},
    "lvis_classes": [],       # List of LVIS class names for label lookup
    "known_class_ids": [],    # List of known class IDs
    "model_versions": {
        "grounding_dino": "Swin-T (GDINO_weights.pth)",
        "mask_rcnn": "R101-FPN New Baseline (MaskRCNN_v2.pt)",
        "sam": "ViT-H (SAM_weights.pth)",
        "clip": "ViT-B-32 (open_clip laion2b_s34b_b79k)",
        "vlrm": "BLIP-2 OPT-2.7B (sashakunitsyn/vlrm-blip2-opt-2.7b)",
    },
    "stage2_processor": None,   # Stage2_VLRM_CLIP_Fusion instance
}

# Finalized 80 Seen (Known) Category IDs for LVIS OVD
KNOWN_COCO_IDS = {
    3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 169, 173, 183,
    207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
    611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
    964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
    1139, 1190, 1202
}


def init_pipeline():
    """Initialize all pipeline models (lazy, first call only)."""
    if STATE["initialized"] or STATE["loading"]:
        return

    STATE["loading"] = True
    logger.info("=" * 60)
    logger.info("Initializing pipeline models...")
    logger.info("=" * 60)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)

        params = {
            "detectron2_dir": os.environ.get("DETECTRON2_DIR", "datasets/DETECTRON2_DATASETS"),
            "gdino_checkpoint": os.environ.get("GDINO_CHECKPOINT", "GDINO_weights.pth"),
            "cfg_file": os.environ.get("CFG_FILE", "cfg/MaskRCNN_R101-FPN-New-Baseline/R101-FPN-New-Baseline.py"),
            "rcnn_weight_dir": os.environ.get("RCNN_WEIGHT_DIR", "maskrcnn_v2"),
            "sam_checkpoint": os.environ.get("SAM_CHECKPOINT", "SAM_weights.pth"),
            "class_len_per_prompt": int(os.environ.get("CLASS_LEN_PER_PROMPT", "81")),
            "lvis_data_split": os.environ.get("LVIS_DATA_SPLIT", "lvis_v1_val")
        }

        det_dir = str(PROJECT_ROOT / params["detectron2_dir"])
        os.environ["DETECTRON2_DATASETS"] = det_dir

        # Fix GroundingDINO compatibility with newer transformers versions
        try:
            import transformers
            from transformers import BertModel, BertPreTrainedModel
            
            # 1. Missing get_head_mask in BertModel/BertPreTrainedModel
            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                return [None] * num_hidden_layers
                
            # Broad attack: patch both class and any potentially loaded instances
            for target in [BertModel, BertPreTrainedModel]:
                if not hasattr(target, "get_head_mask"):
                    target.get_head_mask = get_head_mask
            
            # Check if there are other models in transformers.models.bert.modeling_bert
            try:
                import transformers.models.bert.modeling_bert as bert_mod
                if hasattr(bert_mod, "BertModel") and not hasattr(bert_mod.BertModel, "get_head_mask"):
                    bert_mod.BertModel.get_head_mask = get_head_mask
            except: pass
            
            logger.info("  ✓ Applied BertModel.get_head_mask monkeypatches")
            
            # 2. Monkeypatch get_extended_attention_mask for compatibility with GroundingDINO BertWarper
            import torch
            if not hasattr(BertModel, "_old_geam"):
                BertModel._old_geam = BertModel.get_extended_attention_mask
                def patched_geam(self, attention_mask, input_shape, device=None, dtype=None):
                    # GroundingDINO calls: self.get_extended_attention_mask(attention_mask, input_shape, device)
                    # New API expects: (attention_mask, input_shape, dtype=...)
                    if device is not None and not isinstance(device, torch.dtype):
                        target_dtype = dtype if isinstance(dtype, torch.dtype) else self.dtype
                        return self._old_geam(attention_mask.to(target_dtype), input_shape, dtype=target_dtype)
                    return self._old_geam(attention_mask, input_shape, dtype=device if dtype is None else dtype)
                BertModel.get_extended_attention_mask = patched_geam
                logger.info("  ✓ Applied BertModel.get_extended_attention_mask monkeypatch")

            # Monkeypatch torch.load for PyTorch 2.6 compatibility with legacy weights
            import torch
            original_load = torch.load
            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            logger.info("  ✓ Applied torch.load compatibility monkeypatch for legacy weights")
        except Exception as e:
            logger.warning(f"Could not apply BertModel monkeypatch: {e}")

        import sys
        scripts_dir = str(PROJECT_ROOT / "scripts" / "novel_object_detection")
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)

        from inference_single_image import setup

        outputs_dir = str(PROJECT_ROOT / "outputs" / "api")
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)

        model, text_prompt_list, param_dict = setup(
            outputs_dir,
            str(PROJECT_ROOT / params["gdino_checkpoint"]),
            params["cfg_file"],
            str(PROJECT_ROOT / params["rcnn_weight_dir"]),
            str(PROJECT_ROOT / params["sam_checkpoint"]),
            params["class_len_per_prompt"],
        )

        param_dict["visualize"] = True

        # Load LVIS class names for label resolution
        lvis_split = params.get("lvis_data_split", "lvis_v1_val")
        from detectron2.data import MetadataCatalog
        meta = MetadataCatalog.get(lvis_split)
        lvis_classes = meta.thing_classes if hasattr(meta, "thing_classes") else []
        logger.info(f"Loaded {len(lvis_classes)} LVIS class names")

        # Initialize Stage-2 Semantic Fusion Processor (Mahalanobis Distance)
        try:
            from vlrm_clip_module import Stage2_VLRM_CLIP_Fusion
            stage2_processor = Stage2_VLRM_CLIP_Fusion(
                device="cuda" if torch.cuda.is_available() else "cpu",
                fusion_alpha=0.4
            )
            STATE["stage2_processor"] = stage2_processor
            logger.info("  ✓ Stage-2 VLRM+CLIP Fusion processor initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Stage-2 processor: {e}")
 
        STATE["model"] = model
        STATE["text_prompt_list"] = text_prompt_list
        STATE["param_dict"] = param_dict if param_dict is not None else {}
        STATE["lvis_classes"] = list(lvis_classes) if lvis_classes else [] # Fix type assignment
        STATE["known_class_ids"] = list(KNOWN_COCO_IDS) # Fix type assignment
        STATE["initialized"] = True
        STATE["error"] = None

        if isinstance(STATE["model_versions"], dict):
            for name, ver in STATE["model_versions"].items():
                logger.info(f"  ✓ {name}: {ver}")

        logger.info("Pipeline ready!")

    except Exception as e:
        STATE["error"] = str(e)
        logger.error(f"Pipeline init FAILED: {e}")
        logger.error(traceback.format_exc())
    finally:
        STATE["loading"] = False


from torchvision.ops import batched_nms
from detectron2.structures import Instances, Boxes
import cv2

def postprocess_instances(instances, lvis_classes, known_ids):
    if instances is None or len(instances) == 0:
        return instances
        
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    
    keep_idxs = []
    
    for i in range(len(boxes)):
        cls_id = int(classes[i].item())
        score = float(scores[i].item())
        
        # known vs unknown
        det_type = "known" if (cls_id + 1) in known_ids else "unknown"
        
        # 1. Strict Confidence Filtering
        if det_type == "known" and score < 0.5:
            continue
        if det_type == "unknown" and score < 0.6:
            continue
        if score <= 0.0:
            continue
            
        # 3. Remove Background Predictions Completely
        label = lvis_classes[cls_id] if lvis_classes and 0 <= cls_id < len(lvis_classes) else ""
        label_lower = label.lower()
        if not label_lower or label_lower in ["background", "null", "invalid", "bg"]:
            continue
            
        # 4. Enforce Object Completeness: Tiny or Huge boxes
        img_h, img_w = instances.image_size
        img_area = img_h * img_w
        x1, y1, x2, y2 = boxes[i].tolist()
        box_area = (x2 - x1) * (y2 - y1)
        
        if box_area < 0.001 * img_area: # Less than 0.1% of image
            continue
        if box_area > 0.95 * img_area: # Covering more than 95% of image
            continue
            
        # 5. Remove False Person Detections
        if label_lower == "person" and box_area < 0.03 * img_area and score < 0.75:
            continue
            
        keep_idxs.append(i)
        
    if len(keep_idxs) == 0:
        import torch
        empty_inst = Instances(instances.image_size)
        empty_inst.pred_boxes = Boxes(torch.empty(0, 4, device=boxes.device))
        empty_inst.scores = torch.empty(0, device=scores.device)
        empty_inst.pred_classes = torch.empty(0, dtype=torch.int64, device=classes.device)
        return empty_inst
        
    import torch
    keep_tensor = torch.tensor(keep_idxs, dtype=torch.long, device=boxes.device)
    
    filtered_boxes = boxes[keep_tensor]
    filtered_scores = scores[keep_tensor]
    filtered_classes = classes[keep_tensor]
    
    # 2. Apply Proper Class-wise NMS
    nms_keep = batched_nms(filtered_boxes.float(), filtered_scores.float(), filtered_classes, iou_threshold=0.5)
    
    final_inst = Instances(instances.image_size)
    final_inst.pred_boxes = Boxes(filtered_boxes[nms_keep])
    final_inst.scores = filtered_scores[nms_keep]
    final_inst.pred_classes = filtered_classes[nms_keep]
    
    return final_inst

def instances_to_detections(instances, lvis_classes, known_ids):
    """
    Convert Detectron2 Instances to JSON-serializable detection list.

    Each detection:
      { id, label, confidence, type, bbox }
    """
    detections = []
    if instances is None or len(instances) == 0:
        return detections

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    for i in range(len(boxes)):
        cls_id = int(classes[i])
        score = float(scores[i])

        # Resolve label
        if lvis_classes and 0 <= cls_id < len(lvis_classes):
            label = lvis_classes[cls_id]
        else:
            label = f"class_{cls_id}"

        # Known vs Unknown
        det_type = "known" if (cls_id + 1) in known_ids else "unknown"
        
        # 4. Clean Unknown Labeling & Formatting
        if det_type == "unknown":
            words = label.split()
            if len(words) > 2:
                # Limit caption length
                label = "Unknown Object"
                
        # Title case and remove underscores
        label = label.replace("_", " ").title()
        
        # Extra safeguard against background
        if label.lower() in ["background", "null", "invalid", "bg"]:
            continue

        bbox = boxes[i].tolist()
        bbox = [float(f"{v:.1f}") for v in bbox]

        detections.append({
            "id": i,
            "label": str(label),
            "confidence": float(f"{score:.2f}"),
            "type": str(det_type),
            "bbox": bbox
        })

    return detections


def draw_final_image(img_path, detections):
    """Draw clean final visualization based only on final JSON detections."""
    img = cv2.imread(img_path)
    
    # Sort detections by score (lowest to highest) so highest score is drawn on top
    detections = sorted(detections, key=lambda x: x["confidence"])
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det["confidence"]
        label = det["label"]
        det_type = det["type"]
        
        # Green for known, Orange for unknown
        color = (0, 255, 0) if det_type == "known" else (0, 165, 255) # BGR
        
        # Box thickness = 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        text = f"{label} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw text background (Label above box, padding=3px)
        bg_y1 = y1 - text_height - (padding * 2) - baseline
        bg_y2 = y1
        if bg_y1 < 0: # If label goes out of frame, draw inside box
            bg_y1 = y1
            bg_y2 = y1 + text_height + (padding * 2) + baseline
            text_y = y1 + text_height + padding
        else:
            text_y = y1 - padding - baseline
            
        cv2.rectangle(img, (x1, bg_y1), (x1 + text_width + (padding * 2), bg_y2), color, -1)
        
        # Draw text in white
        cv2.putText(img, text, (x1 + padding, text_y), font, font_scale, (255, 255, 255), thickness)
        
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')
    """Read an image file and return base64 string."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    if STATE["initialized"]:
        status = "online"
    elif STATE["loading"]:
        status = "loading"
    elif STATE["error"]:
        status = "error"
    else:
        status = "idle"

    return {
        "status": status,
        "error": STATE["error"],
        "models": STATE["model_versions"],
    }


@app.get("/api/models")
async def models():
    return {
        "models": STATE["model_versions"],
        "initialized": STATE["initialized"],
    }


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """Run the full 3-stage pipeline on an uploaded image."""

    # 1. Ensure pipeline is loaded
    if not STATE["initialized"]:
        logger.info("Detect requested but pipeline not initialized. Initializing now...")
        init_pipeline()
    
    if STATE["error"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization failed: {STATE['error']}",
        )

    # 2. Read uploaded image
    try:
        raw = await file.read()
        pil_img = Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    # 3. Save to temp file (pipeline reads from filesystem)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="/tmp") as tmp:
            pil_img.save(tmp, format="JPEG", quality=95)
            tmp_path = tmp.name

        logger.info(f"Running detection on {tmp_path} ({pil_img.size[0]}x{pil_img.size[1]})")

        # 4. Import pipeline functions
        import torch
        import sys
        
        # Add scripts dir to path to import inference tools
        scripts_dir = str(PROJECT_ROOT / "scripts" / "novel_object_detection")
        if scripts_dir not in sys.path:
            sys.path.append(scripts_dir)
            
        from evaluation import inference_single_image as run_inference
        from utils import read_image
        import detectron2.data.transforms as T

        model = STATE.get("model")
        text_prompt_list = STATE.get("text_prompt_list")
        
        # Explicitly initialize and type param_dict to avoid Pyre assignment errors
        raw_params = STATE.get("param_dict")
        param_dict: Dict[str, Any] = {}
        if isinstance(raw_params, dict):
            for k, v in raw_params.items():
                param_dict[str(k)] = v
        
        # Merge model and text_prompt_list into param_dict for Stage 1
        param_dict["gdino_model"] = model
        param_dict["text_prompt_list"] = text_prompt_list
        
        # Set output dir
        out_dir = str(PROJECT_ROOT / "outputs" / "api")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        param_dict["out_dir"] = out_dir
        param_dict["visualize"] = True

        # 5. Prepare input (same as inference_single_image.py)
        img = read_image(tmp_path, format="BGR")
        orig_h, orig_w = img.shape[0], img.shape[1]

        data_dict = {
            "file_name": os.path.abspath(tmp_path),
            "height": orig_h,
            "width": orig_w,
            "not_exhaustive_category_ids": [],
            "neg_category_ids": [],
            "image_id": 0,
        }

        augmentations = T.AugmentationList([
            T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        ])
        aug_input = T.AugInput(img, sem_seg=None)
        augmentations(aug_input)
        data_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(aug_input.image.transpose(2, 0, 1))
        )

        inputs = [data_dict]

        # 6. Run full pipeline but break it down to get stage-wise images
        t_start = time.time()

        from ground_dino_utils import run_stage1_inference, run_stage3_inference
        from utils import BBoxVisualizer
        from detectron2.data import MetadataCatalog
        import cv2
        
        # We need to temporarily re-implement `inference_gdino` here to intercept stages, 
        # or just run them sequentially as done in ground_dino_utils.py.
        # Ensure core models are on GPU for Stage 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "gdino_model" in param_dict:
            param_dict["gdino_model"].to(device)
        if "rcnn_model" in param_dict:
            param_dict["rcnn_model"].to(device)

        # Run Stage 1
        s1_out = run_stage1_inference(inputs, param_dict)
        # Generate Stage 1 Image Block (Removed for cleanliness)
        stage1_time = time.time() - t_start
        
        # Move Stage 1 models back to CPU to save memory for Stage 2
        if "gdino_model" in param_dict:
            param_dict["gdino_model"].to("cpu")
        if "rcnn_model" in param_dict:
            param_dict["rcnn_model"].to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        # Run Stage 2: Unknown Object Labeling
        t_s2_start = time.time()
        bg_boxes = s1_out["background"]["boxes"]
        
        stage2_out = {
            "boxes": torch.empty(0, 4),
            "scores": torch.empty(0),
            "labels": torch.empty(0)
        }
        
        if len(bg_boxes) > 0 and STATE["stage2_processor"] is not None:
            logger.info(f"[Stage-2] Processing {len(bg_boxes)} background proposals...")
            # Use the integrated module for Stage 2 (VLRM + CLIP + Mahalanobis)
            s2_scores_raw, s2_valid_idxs = STATE["stage2_processor"].run_stage_2(
                background_rois=bg_boxes,
                full_image=pil_img,
                class_names=STATE["lvis_classes"],
                max_regions=150, # High discovery for web app to capture everything
                min_crop_size=10,
                keep_on_gpu=False # Move VLRM back to CPU to save VRAM for Stage 3
            )
            # Explicitly clear VRAM after VLRM
            torch.cuda.empty_cache()
            gc.collect()
            
            if len(s2_valid_idxs) > 0:
                s2_best_scores, s2_best_labels = s2_scores_raw.max(dim=1)
                
                # Dynamic Thresholding for Unknown Classification Boundary
                s2_mean = s2_scores_raw.mean(dim=0)
                s2_std = s2_scores_raw.std(dim=0)
                dynamic_thresh = s2_mean + 2 * s2_std
                
                # Check if top score meets the dynamic boundary distribution for that class
                # If it fails, that means it's an UNKNOWN object not matching an LVIS category!
                fails_threshold = s2_best_scores <= dynamic_thresh[s2_best_labels]
                
                if "Unknown Object" not in STATE["lvis_classes"]:
                    STATE["lvis_classes"].append("Unknown Object")
                unknown_idx = STATE["lvis_classes"].index("Unknown Object")
                
                s2_best_labels[fails_threshold] = unknown_idx
                
                # Base similarity filter for extremely weak background noise
                s2_mask = s2_best_scores >= 0.1
                
                if s2_mask.any():
                    stage2_out = {
                        "boxes": bg_boxes[s2_valid_idxs][s2_mask],
                        "scores": s2_best_scores[s2_mask],
                        "labels": s2_best_labels[s2_mask]
                    }
                    logger.info(f"[Stage-2] Preserved {s2_mask.sum()} novel candidates ({fails_threshold.sum()} marked strictly unknown)")
        
        stage2_time = time.time() - t_s2_start

        # Run Stage 3: Fusion and Refinement
        t_s3_start = time.time()
        
        # Ensure SAM is on GPU for Stage 3
        if "sam" in param_dict:
            param_dict["sam"].to(device)
            
        final_outputs = run_stage3_inference(s1_out, stage2_out, param_dict)
        
        # Move SAM to CPU after Stage 3 to clear VRAM
        if "sam" in param_dict:
            param_dict["sam"].to("cpu")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        stage3_time = time.time() - t_s3_start

        t_total = time.time() - t_start
        logger.info(f"Pipeline done in {t_total:.2f}s (S1={stage1_time:.2f}s, S2={stage2_time:.2f}s, S3={stage3_time:.2f}s)")

        # 7. Post-process Instances
        instances = final_outputs[0]["instances"] if final_outputs else None
        lvis_classes = STATE.get("lvis_classes", [])
        known_ids = STATE.get("known_class_ids", set())
        
        instances = postprocess_instances(instances, lvis_classes, known_ids)

        detections = instances_to_detections(instances, lvis_classes, known_ids)
        logger.info(f"Extracted {len(detections)} refined detections")

        # 8. Create Final Image
        annotated_b64 = draw_final_image(tmp_path, detections)

        # 9. Count known/unknown
        known_count = sum(1 for d in detections if d["type"] == "known")
        unknown_count = sum(1 for d in detections if d["type"] == "unknown")

        # Remove intermediate stages from the payload 
        response = {
            "detections": detections,
            "annotated_image": annotated_b64,
            "stage_times": {
                "stage1": float(f"{stage1_time:.2f}"),
                "stage2": float(f"{stage2_time:.2f}"),
                "stage3": float(f"{stage3_time:.2f}"),
                "total": float(f"{t_total:.2f}")
            },
            "metrics": {
                "total_detections": len(detections),
                "known_count": known_count,
                "novel_count": unknown_count,
                "processing_time": float(f"{t_total:.2f}"),
            },
        }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DETECTION FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "detections": [],
            "debug": {"traceback": traceback.format_exc()}
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass


@app.on_event("startup")
async def on_startup():
    logger.info("=" * 60)
    logger.info("Cooperative Novel Object Detection API")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info("Models loaded lazily on first /api/detect request")
    logger.info("=" * 60)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
