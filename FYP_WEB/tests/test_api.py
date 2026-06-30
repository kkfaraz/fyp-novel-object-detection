#!/usr/bin/env python3
import os
import sys
import io
import cv2
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.server import app

client = TestClient(app)

def create_mock_image() -> bytes:
    # Create a simple 100x100 RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 0, 0), -1)
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()

def test_health_endpoint():
    print("Testing /api/health...")
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models" in data
    print("✓ /api/health passed!")

def test_models_endpoint():
    print("Testing /api/models...")
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "initialized" in data
    print("✓ /api/models passed!")

def test_detect_endpoint():
    print("Testing /api/detect (Mock validation)...")
    img_bytes = create_mock_image()
    
    # We send request to /api/detect
    # Note: To prevent long-running model loading during mock test,
    # we can mock the pipeline coordinator or run it end-to-end if weights are loaded.
    # Since server.py calls pipeline.run inside detect(), we can mock it here to verify API schema.
    from unittest.mock import MagicMock
    import backend.api.server as server_module
    
    # Mock pipeline instance
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = {
        "detections": [
            {
                "id": 0,
                "label": "mock_object",
                "confidence": 0.95,
                "type": "known",
                "bbox": [10.0, 10.0, 90.0, 90.0],
                "detector_source": "mock"
            }
        ],
        "annotated_image": "mock_base64_string",
        "stage_images": {
            "stage1": "mock_s1",
            "stage2": "mock_s2",
            "stage3": "mock_s3"
        },
        "stage_times": {
            "stage1": 0.1,
            "stage2": 0.2,
            "stage3": 0.3,
            "total": 0.6
        },
        "metrics": {
            "total_detections": 1,
            "known_count": 1,
            "novel_count": 0,
            "processing_time": 0.6
        },
        "known_objects": [],
        "novel_objects": [],
        "debug": {
            "image_size": [100, 100],
            "pipeline_version": "mock"
        }
    }
    
    server_module.STATE["pipeline"] = mock_pipeline
    server_module.STATE["initialized"] = True
    
    files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/detect?dataset=coco", files=files)
    
    assert response.status_code == 200
    res_data = response.json()
    assert "detections" in res_data
    assert len(res_data["detections"]) == 1
    assert res_data["detections"][0]["label"] == "mock_object"
    assert "annotated_image" in res_data
    assert "stage_times" in res_data
    assert "metrics" in res_data
    print("✓ /api/detect schema validation passed!")

if __name__ == "__main__":
    test_health_endpoint()
    test_models_endpoint()
    test_detect_endpoint()
    print("=" * 60)
    print("All API routes tests completed successfully!")
    print("=" * 60)
