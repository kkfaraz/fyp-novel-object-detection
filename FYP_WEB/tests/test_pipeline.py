#!/usr/bin/env python3
import os
import sys
import tempfile
import torch
import cv2
import numpy as np
import time
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline_coordinator import CooperativeDetectionPipeline

def create_dummy_image(path: str):
    # Create a 640x480 gray image with some shapes
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    # Draw a rectangle representing an object
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)
    cv2.imwrite(path, img)

def test_pipeline():
    print("=" * 60)
    print("Running Cooperative Detection Pipeline End-to-End Test")
    print("=" * 60)

    # 1. Create temp image in workspace
    temp_dir = PROJECT_ROOT / "backend" / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_img_path = temp_dir / "test_dummy.jpg"
    create_dummy_image(str(temp_img_path))
    print(f"Created dummy test image at: {temp_img_path}")

    # 2. Instantiate pipeline
    try:
        pipeline = CooperativeDetectionPipeline()
        print("Pipeline successfully instantiated.")
    except Exception as e:
        print(f"Pipeline instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. Test on different strategies
    strategies = ["coco", "voc", "lvis"]
    for strat in strategies:
        print(f"\n--- Testing strategy: {strat.upper()} ---")
        try:
            t0 = time.time() if "time" in globals() else 0
            # Mock kwargs to speed up testing
            result = pipeline.run(
                str(temp_img_path),
                dataset_name=strat,
                clip_threshold=0.5,
                wbf_iou_threshold=0.5
            )
            
            # 4. Validate output schema
            assert "detections" in result, "Missing 'detections' key in output."
            assert "annotated_image" in result, "Missing 'annotated_image' key."
            assert "stage_images" in result, "Missing 'stage_images' key."
            assert "stage_times" in result, "Missing 'stage_times' key."
            assert "metrics" in result, "Missing 'metrics' key."
            
            print(f"✓ {strat.upper()} Strategy execution succeeded!")
            print(f"  - Detections found: {len(result['detections'])}")
            print(f"  - Stage times: {result['stage_times']}")
            print(f"  - Metrics: {result['metrics']}")

        except Exception as e:
            print(f"✗ Strategy {strat.upper()} failed: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup temp image
    if temp_img_path.exists():
        temp_img_path.unlink()
        print("\nCleaned up dummy test image.")

if __name__ == "__main__":
    test_pipeline()
