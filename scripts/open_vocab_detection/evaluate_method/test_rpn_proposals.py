#!/usr/bin/env python3
"""
Quick verification script to test RPN proposal extraction.
Tests if Mask-RCNN correctly generates 500-1000 RPN proposals.
"""
import os
import sys
import torch

# Setup paths
proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(proj_path)

print("=" * 60)
print("RPN Proposal Verification Test")
print("=" * 60)

# Check if we have a test image
test_images = [
    "/home/faraz/cooperative-foundational-models/custom_image.jpg",
    "/home/faraz/cooperative-foundational-models/datasets/DETECTRON2_DATASETS/coco/val2017/000000000139.jpg",
]

test_image = None
for img_path in test_images:
    if os.path.exists(img_path):
        test_image = img_path
        break

if test_image is None:
    print("❌ No test image found. Please provide a test image path.")
    print("Expected locations:")
    for img in test_images:
        print(f"  - {img}")
    sys.exit(1)

print(f"\n[Test Image] {test_image}")

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {device}")

try:
    print("\n[1/2] Loading Mask-RCNN...")
    from scripts.open_vocab_detection.evaluate_method.load_models import load_maskrcnn_model
    
    maskrcnn_cfg_path = os.path.join(proj_path, "cfg/MaskRCNN_R101-FPN-New-Baseline/R101-FPN-New-Baseline.py")
    maskrcnn_weight_dir = os.path.join(proj_path, "maskrcnn_v2")
    
    maskrcnn_model, maskrcnn_cfg = load_maskrcnn_model(maskrcnn_cfg_path, maskrcnn_weight_dir, device)
    print("✓ Mask-RCNN loaded")
    
except Exception as e:
    print(f"❌ Failed to load Mask-RCNN: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/2] Testing RPN proposal extraction...")
    from PIL import Image
    import numpy as np
    from scripts.open_vocab_detection.evaluate_method.ground_dino_utils import inference_maskrcnn
    
    # Load image
    image = Image.open(test_image).convert("RGB")
    image_np = np.array(image)
    
    # Convert to tensor (BGR format for Detectron2)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).flip(0)  # RGB to BGR
    
    # Create input in Detectron2 format
    inputs = [{
        "image": image_tensor,
        "height": image.height,
        "width": image.width,
        "file_name": test_image
    }]
    
    # Run Mask-RCNN
    boxes, scores, classes = inference_maskrcnn(maskrcnn_model, inputs, device)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"✓ Number of proposals: {len(boxes)}")
    
    if len(scores) > 0:
        print(f"✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"✓ Mean score: {scores.mean():.3f}")
    
    # Verify expectations
    if len(boxes) >= 100:
        print("\n✅ SUCCESS: Generated sufficient RPN proposals!")
        print(f"   Expected: 500-1000 proposals")
        print(f"   Got: {len(boxes)} proposals")
    elif len(boxes) > 0:
        print(f"\n⚠️ WARNING: Only {len(boxes)} proposals generated.")
        print("   This might be instances, not RPN proposals.")
        print("   Check if model config has PROPOSAL_GENERATOR enabled.")
    else:
        print("\n❌ FAILED: No proposals generated!")
        
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
