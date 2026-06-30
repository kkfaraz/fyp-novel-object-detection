import os
import sys
import time
import logging
import traceback
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Setup sys.path to resolve backend package correctly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup temporary directory inside workspace
TEMP_DIR = PROJECT_ROOT / "backend" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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

# ─── Global State ─────────────────────────────────────────────────────────────
STATE: Dict[str, Any] = {
    "initialized": False,
    "loading": False,
    "error": None,
    "pipeline": None,
    "model_versions": {
        "grounding_dino": "Swin-T (GDINO_weights.pth)",
        "mask_rcnn": "R101-FPN New Baseline (MaskRCNN_v2.pt)",
        "sam": "ViT-H (SAM_weights.pth)",
        "clip": "ViT-SO400M-14-SigLIP (webli)",
    }
}

def init_pipeline():
    """Lazy initializer for the cooperative detection pipeline."""
    if STATE["initialized"] or STATE["loading"]:
        return

    STATE["loading"] = True
    logger.info("=" * 60)
    logger.info("Initializing Cooperative Detection Pipeline...")
    logger.info("=" * 60)

    try:
        from backend.services.pipeline_coordinator import CooperativeDetectionPipeline
        STATE["pipeline"] = CooperativeDetectionPipeline()
        STATE["initialized"] = True
        STATE["error"] = None
        logger.info("Pipeline coordinator and model weights successfully loaded.")
    except Exception as e:
        STATE["error"] = str(e)
        logger.error(f"Pipeline initialization failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        STATE["loading"] = False

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
async def detect(
    file: UploadFile = File(...),
    dataset: str = Query("lvis", description="Dataset strategy to use (coco, voc, lvis)")
):
    """Runs the cooperative detection pipeline on an uploaded image."""
    
    # 1. Ensure pipeline is loaded
    if not STATE["initialized"]:
        logger.info("First request received. Initializing pipeline...")
        init_pipeline()
        
    if STATE["error"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model initialization failed: {STATE['error']}",
        )

    # 2. Read and save uploaded image to local workspace-safe temp path
    tmp_path = None
    try:
        raw_bytes = await file.read()
        pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        
        # Safe temp filename
        safe_name = f"upload_{int(time.time() * 1000)}.jpg"
        tmp_path = TEMP_DIR / safe_name
        
        pil_img.save(tmp_path, format="JPEG", quality=95)
        logger.info(f"Running detection on {tmp_path} ({pil_img.size[0]}x{pil_img.size[1]}) using {dataset} strategy")

        # 3. Execute cooperative pipeline
        pipeline = STATE["pipeline"]
        result = pipeline.run(str(tmp_path), dataset_name=dataset)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"DETECTION FATAL ERROR: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "detections": [],
                "debug": {"traceback": traceback.format_exc()}
            }
        )
    finally:
        # Cleanup uploaded temp file
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

# ─── Serve Frontend Static Files ──────────────────────────────────────────────
# Resolve frontend build directory path
frontend_dir = PROJECT_ROOT / "frontend" / "dist"
if not frontend_dir.exists():
    frontend_dir = PROJECT_ROOT / "FYP_WEB" / "frontend" / "dist"

if frontend_dir.exists():
    logger.info(f"Mounting frontend static files from: {frontend_dir}")
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
else:
    logger.warning(f"Frontend static files directory not found at: {frontend_dir}")

@app.on_event("startup")
async def on_startup():
    logger.info("=" * 60)
    logger.info("Cooperative Novel Object Detection API Server Starting")
    logger.info(f"Workspace Root: {PROJECT_ROOT}")
    logger.info("=" * 60)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
