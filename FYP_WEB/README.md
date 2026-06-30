# Production-Quality Cooperative Novel Object Detection Framework (FYP_WEB)

This repository contains the upgraded, modularized, and production-grade backend for the **Multi-Dataset Cooperative Novel Object Detection Framework**. It integrates deep supervised models (Mask R-CNN), open-vocabulary zero-shot detectors (Grounding DINO), contrastive vision-language verification (CLIP), segment-guided boundary refinement (SAM), and confidence calibration strategies into a cohesive, high-performance API service.

---

## 🚀 Key Improvements & Architecture

- **Context-Aware Portability**: Dynamically detects runtime environments (Local, Google Colab, Kaggle) to automatically scale CPU/GPU resources and handle legacy weight path routing.
- **Model Registry & Lazy Loading**: Caches heavyweight neural network models (`ModelRegistry`) and applies runtime monkeypatches for `BertModel` and `torch.load` to ensure seamless loading.
- **Dataset Strategy Pattern**: Encapsulates class names, prompt templating, synonyms mapping, and score thresholds for **COCO**, **VOC**, and **LVIS** datasets.
- **Unified Detection Adapters**: Standardizes predictions from heterogeneous model outputs (classes, coordinates, confidence, and source provenance attributes).
- **Confidence Calibration**: Implements Temperature Scaling, Platt Scaling, and Isotonic Regression configurations to map raw model logits to calibrated detection probabilities.
- **Adaptive Proposal Fusion**: Merges overlapping boxes from supervised and zero-shot detectors using Weighted Box Fusion (WBF) and Soft-NMS.
- **Semantic Crop Verification**: Employs CLIP (SigLIP ViT-SO400M-14) to verify unsupervised region proposal crops against target dataset vocabularies.
- **SAM Boundary Refinement**: Runs Segment Anything (SAM) on bounding box coordinates to produce pixel-tight, refined object boundaries.
- **Label Mapping API**: Sanitizes and translates category spelling mismatches between model catalogs and React UI structures using a local `label_mapping.json` schema.
- **Multi-threaded Execution Engine**: Concurrently executes inference stages using thread pools when hardware resources allow, fallback to sequential mode on lightweight Colab/Kaggle instances.

---

## 📁 Directory Structure

```directory
FYP_WEB/
├── Dockerfile                  # Container build recipe
├── docker-compose.yml          # GPU-enabled container services orchestrator
├── run_local.py                # Local environment API server runner
├── run_colab.py                # Google Colab API runner (with ngrok tunnel)
├── run_kaggle.py               # Kaggle environment API runner
├── .env.example                # Template for configuration environment variables
├── requirements.txt            # Python package dependencies
├── frontend/                   # React Frontend App (Unchanged)
└── backend/
    ├── api/
    │   └── server.py           # FastAPI server endpoints
    ├── config/
    │   ├── label_mapping.json  # UI spelling translator mapping
    │   └── cfg/                # Detectron2 & Grounding DINO configs
    └── services/
        ├── adapters/           # Dataset adapters (standardizes class & box lists)
        ├── calibration/        # Platt/Isotonic/Temperature calibrators
        ├── detectors/          # Model wrapper adapters (GDINO, Mask R-CNN)
        ├── engine/             # Sequential/Threaded execution coordinator
        ├── fusion/             # Weighted Box Fusion (WBF) & Soft-NMS
        ├── label_mapping/      # Mapped label naming lookups
        ├── pipeline_coordinator.py # E2E Orchestrator (Stages 1, 2, and 3)
        ├── refinement/         # SAM boundary refiner & CLIP verifier
        ├── registry/           # Lazy model registry loader
        ├── strategies/         # Strategy patterns (COCO, VOC, LVIS config)
        └── utils/              # Env managers, prompt helpers, visualizers
```

---

## ⚡ Setup & Execution

### 1. Local execution
First, activate your virtual environment, then run the startup script:
```bash
python run_local.py
```
The server will start on `http://localhost:8000`.

### 2. Google Colab execution
1. Upload the `FYP_WEB/` folder to your Google Drive.
2. Mount your Google Drive and set the `NGROK_AUTHTOKEN` environment variable.
3. Run the runner script:
```bash
python run_colab.py
```
This automatically establishes a public secure ngrok tunnel and outputs the URL to hook into your frontend application.

### 3. Docker Compose (GPU-accelerated)
Run the entire application in a container with full access to host GPUs:
```bash
docker-compose up --build
```

---

## 🔌 API Documentation

### 1. Get Health Status
* **Endpoint**: `GET /api/health`
* **Response**:
```json
{
  "status": "online",
  "error": null,
  "models": {
    "grounding_dino": "Swin-T (GDINO_weights.pth)",
    "mask_rcnn": "R101-FPN New Baseline (MaskRCNN_v2.pt)",
    "sam": "ViT-H (SAM_weights.pth)",
    "clip": "ViT-SO400M-14-SigLIP (webli)"
  }
}
```

### 2. Run Novel Object Detection
* **Endpoint**: `POST /api/detect`
* **Parameters**:
  - `file`: (Multipart Form File) Uploaded Image.
  - `dataset`: (Query String, default `lvis`) Target vocabulary strategy (`coco`, `voc`, `lvis`).
* **Response**:
```json
{
  "detections": [
    {
      "id": 0,
      "label": "person",
      "confidence": 0.9421,
      "type": "known",
      "bbox": [102.1, 45.4, 250.3, 400.0],
      "detector_source": "maskrcnn_known"
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "stage_images": {
    "stage1": "data:image/jpeg;base64,...",
    "stage2": "data:image/jpeg;base64,...",
    "stage3": "data:image/jpeg;base64,..."
  },
  "stage_times": {
    "stage1": 1.25,
    "stage2": 0.42,
    "stage3": 0.88,
    "total": 2.55
  },
  "metrics": {
    "total_detections": 1,
    "known_count": 1,
    "novel_count": 0,
    "processing_time": 2.55
  }
}
```

---

## 🧪 Testing & Verification
We supply automated end-to-end tests checking multi-strategy loading, WBF processing, and image response formatting:
```bash
python tests/test_pipeline.py
```
