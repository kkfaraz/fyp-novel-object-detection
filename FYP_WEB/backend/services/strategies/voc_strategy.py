from typing import List, Dict, Any, Tuple
import torch
from backend.services.strategies.base import DatasetStrategy
from backend.services.utils.prompt_utils import get_text_prompt_list_for_g_dino

class VOCStrategy(DatasetStrategy):
    def __init__(self):
        self.classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", 
            "train", "tvmonitor"
        ]

    def get_dataset_name(self) -> str:
        return "voc"

    def get_classes(self) -> List[str]:
        return self.classes

    def get_prompts(self, tokenizer, class_len_per_prompt: int = 80) -> Tuple[List[str], List[torch.Tensor]]:
        return get_text_prompt_list_for_g_dino(self.classes, tokenizer, class_len_per_prompt)

    def get_thresholds(self) -> Dict[str, float]:
        return {
            "box_threshold": 0.30,  # Slightly lower box threshold for VOC zero-shot generalization
            "text_threshold": 0.20,
            "nms_threshold": 0.50
        }

    def get_calibration_config(self) -> Dict[str, Any]:
        return {
            "method": "identity",
            "params": {}
        }

    def get_label_mapping(self) -> Dict[str, str]:
        # Map VOC names to canonical names
        return {
            "aeroplane": "airplane",
            "diningtable": "dining table",
            "motorbike": "motorcycle",
            "pottedplant": "potted plant",
            "tvmonitor": "tv"
        }
