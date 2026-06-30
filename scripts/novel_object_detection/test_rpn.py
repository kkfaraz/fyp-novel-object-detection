"""Quick diagnostic: test RPN extraction from Mask-RCNN on a single LVIS image."""
import os
import sys
import json
import torch
import cv2

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
proj_path = os.path.abspath(os.path.join(script_dir, "../../"))
sys.path.insert(0, os.path.join(script_dir, '..'))

params_path = os.path.join(script_dir, "params.json")
with open(params_path, 'r') as f:
    params = json.load(f)

detectron2_dir = os.path.join(proj_path, params.get("detectron2_dir", "./datasets/DETECTRON2_DATASETS"))
os.environ['DETECTRON2_DATASETS'] = detectron2_dir

from load_models import load_fully_supervised_trained_model
from detectron2.data import get_detection_dataset_dicts

# Load model
print("Loading Mask-RCNN...")
rcnn_model, cfg = load_fully_supervised_trained_model(params["cfg_file"], params["rcnn_weight_dir"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Model device: {next(rcnn_model.parameters()).device}")
print(f"Model type: {type(rcnn_model)}")

# Load one image
data_split = params.get("data_split", params.get("lvis_data_split", "lvis_v1_val"))
dataset_dicts = get_detection_dataset_dicts(names=data_split, filter_empty=False)
sample = dataset_dicts[0]
print(f"\nTest image: {sample['file_name']}")
print(f"Image size: {sample.get('height')}x{sample.get('width')}")

image = cv2.imread(sample["file_name"])
if image is None:
    print("ERROR: Cannot read image!")
    sys.exit(1)

h, w = image.shape[:2]
image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
print(f"Image tensor shape: {image_tensor.shape}, dtype: {image_tensor.dtype}")

inp = [{"image": image_tensor, "height": h, "width": w, "file_name": sample["file_name"]}]

# Test 1: Full forward pass
print("\n--- Test 1: Full Mask-RCNN forward pass ---")
rcnn_model.eval()
with torch.no_grad():
    outputs = rcnn_model(inp)
    instances = outputs[0]["instances"]
    print(f"Total detections: {len(instances)}")
    print(f"Pred classes: {instances.pred_classes[:10]}")
    print(f"Scores: {instances.scores[:10]}")
    
    known_mask = instances.pred_classes < 80
    print(f"Known (class<80): {known_mask.sum().item()}")
    print(f"Background (class>=80): {(~known_mask).sum().item()}")

# Test 2: RPN proposals
print("\n--- Test 2: RPN proposal_generator ---")
with torch.no_grad():
    images = rcnn_model.preprocess_image(inp)
    print(f"Preprocessed image tensor shape: {images.tensor.shape}, device: {images.tensor.device}")
    
    features = rcnn_model.backbone(images.tensor)
    print(f"Features keys: {list(features.keys())}")
    
    proposals, _ = rcnn_model.proposal_generator(images, features, None)
    print(f"Number of proposal sets: {len(proposals)}")
    
    if len(proposals) > 0:
        p = proposals[0]
        print(f"Proposals type: {type(p)}")
        print(f"Proposals len: {len(p)}")
        print(f"Proposals fields: {p.get_fields().keys()}")
        
        if hasattr(p, 'proposal_boxes'):
            boxes = p.proposal_boxes.tensor
            print(f"Proposal boxes shape: {boxes.shape}")
            print(f"First 5 boxes: {boxes[:5]}")
            
        if hasattr(p, 'objectness_logits'):
            logits = p.objectness_logits
            print(f"Objectness logits shape: {logits.shape}")
            print(f"First 5 logits: {logits[:5]}")
            print(f"First 5 sigmoid: {logits[:5].sigmoid()}")
    else:
        print("NO PROPOSALS GENERATED!")

# Test 3: Check Stage 1 saved data
print("\n--- Test 3: Check saved Stage 1 data ---")
import pickle
s1_path = os.path.join(proj_path, "outputs_lvis/experiments/my_lvis_experiment/stage1_outputs.pkl")
if os.path.exists(s1_path):
    with open(s1_path, 'rb') as f:
        s1_data = pickle.load(f)
    
    sample_s1 = s1_data[0]
    print(f"Stage 1 keys: {sample_s1.keys()}")
    print(f"background keys: {sample_s1['background'].keys()}")
    bg = sample_s1['background']['boxes']
    print(f"Background boxes shape: {bg.shape}, len: {len(bg)}")
    print(f"Known boxes shape: {sample_s1['known']['boxes'].shape}")
    
    # Count how many have non-empty bg
    non_empty = sum(1 for s in s1_data[:100] if len(s['background']['boxes']) > 0)
    print(f"Non-empty bg (first 100): {non_empty}/100")
else:
    print(f"Stage 1 checkpoint not found at: {s1_path}")

print("\nDone!")
