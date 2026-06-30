import sys
import os
import torch
import numpy as np
import time

# Add paths to make sure sam3 and scripts can be imported
sys.path.insert(0, "/home/faraz/FYP/NOD/FYP_CAP_2/sam3")
sys.path.insert(0, "/home/faraz/FYP/NOD/FYP_CAP_2/scripts")

from sam3.model_builder import build_sam3_image_model
from sam3_helper import SAM3CallableWrapper

# Create dummy input
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading model...")
checkpoint_path = "/home/faraz/FYP/NOD/FYP_CAP_2/SAM_weights/sam3.pt"
sam3_model = build_sam3_image_model(
    device=device,
    load_from_HF=False,
    checkpoint_path=checkpoint_path,
    enable_inst_interactivity=True
)

print("Wrapping model...")
sam3_wrapped = SAM3CallableWrapper(sam3_model, device)

print("Preparing dummy image and boxes...")
# Dummy image of size 3x800x800, range [0, 255]
curr_image_tensor = torch.randint(0, 256, (3, 800, 800), dtype=torch.uint8, device=device)
# Dummy boxes (2 boxes, each [x1, y1, x2, y2])
boxes_tensor = torch.tensor([[100.0, 100.0, 300.0, 300.0], [200.0, 200.0, 400.0, 400.0]], dtype=torch.float32, device=device)

# Scale boxes to 1008-resized space like the pipeline does
from segment_anything.utils.transforms import ResizeLongestSide
img_size = sam3_wrapped.image_encoder.img_size
print("SAM3 image size:", img_size)
resize_transform = ResizeLongestSide(img_size)

# Apply resize to image and boxes
img_shape = (800, 800)
curr_image_np = np.random.randint(0, 256, (800, 800, 3), dtype=np.uint8)
curr_image_tensor = resize_transform.apply_image(curr_image_np)
curr_image_tensor = torch.as_tensor(curr_image_tensor, device=device).permute(2, 0, 1).contiguous()
sam_box_prompts = resize_transform.apply_boxes_torch(boxes_tensor, img_shape)

print("Running wrapped SAM3 inference...")
batched_input = [{
    "image": curr_image_tensor,
    "boxes": sam_box_prompts,
    "original_size": img_shape
}]

try:
    with torch.no_grad():
        outputs = sam3_wrapped(batched_input, multimask_output=False)
    print("Inference completed successfully!")
    print("Output keys:", outputs[0].keys())
    print("masks shape:", outputs[0]["masks"].shape)
    print("iou_predictions shape:", outputs[0]["iou_predictions"].shape)
except Exception as e:
    import traceback
    print("Error during inference:")
    traceback.print_exc()
