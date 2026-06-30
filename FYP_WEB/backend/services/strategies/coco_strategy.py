import pickle
from typing import List, Dict, Any, Tuple
import torch
from backend.services.strategies.base import DatasetStrategy
from backend.services.utils.env_manager import EnvironmentManager
from backend.services.utils.prompt_utils import get_text_prompt_list_for_g_dino

class COCOStrategy(DatasetStrategy):
    def __init__(self):
        root = EnvironmentManager.get_project_root()
        pkl_path = root / "backend" / "config" / "coco_ovd_class_to_synonyms.pkl"
        
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                synonyms = pickle.load(f)
            self.classes = sorted(list(synonyms.keys()))
        else:
            # Fallback to standard COCO 80 classes if pickle not found
            self.classes = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
                'toothbrush'
            ]

    def get_dataset_name(self) -> str:
        return "coco"

    def get_classes(self) -> List[str]:
        return self.classes

    def get_prompts(self, tokenizer, class_len_per_prompt: int = 80) -> Tuple[List[str], List[torch.Tensor]]:
        return get_text_prompt_list_for_g_dino(self.classes, tokenizer, class_len_per_prompt)

    def get_thresholds(self) -> Dict[str, float]:
        return {
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "nms_threshold": 0.50
        }

    def get_calibration_config(self) -> Dict[str, Any]:
        # Calibration config for Temperature Scaling, Platt, or Isotonic
        return {
            "method": "identity",
            "params": {}
        }

    def get_label_mapping(self) -> Dict[str, str]:
        return {}
