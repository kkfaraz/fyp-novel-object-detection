import pickle
from typing import List, Dict, Any, Tuple
import torch
from backend.services.strategies.base import DatasetStrategy
from backend.services.utils.env_manager import EnvironmentManager
from backend.services.utils.prompt_utils import get_text_prompt_list_for_g_dino

class LVISStrategy(DatasetStrategy):
    def __init__(self):
        root = EnvironmentManager.get_project_root()
        pkl_path = root / "backend" / "config" / "lvis_original_class_to_synonyms.pkl"
        
        self.classes = []
        
        # Priority 1: Detectron2 MetadataCatalog (ensures correct index alignment)
        try:
            from detectron2.data import MetadataCatalog
            meta = MetadataCatalog.get("lvis_v1_val")
            self.classes = meta.get("thing_classes", [])
        except Exception:
            self.classes = []

        # Priority 2: Fallback to pickle synonyms keys
        if not self.classes and pkl_path.exists():
            with open(pkl_path, "rb") as f:
                synonyms = pickle.load(f)
            # The order must match the catalog's order
            self.classes = list(synonyms.keys())

        # Fallback 3: Hardcoded mini LVIS classes just in case
        if not self.classes:
            self.classes = ["person", "bicycle", "car"] # Minimum safe fallback

    def get_dataset_name(self) -> str:
        return "lvis"

    def get_classes(self) -> List[str]:
        return self.classes

    def get_prompts(self, tokenizer, class_len_per_prompt: int = 81) -> Tuple[List[str], List[torch.Tensor]]:
        return get_text_prompt_list_for_g_dino(self.classes, tokenizer, class_len_per_prompt)

    def get_thresholds(self) -> Dict[str, float]:
        return {
            "box_threshold": 0.35,
            "text_threshold": 0.25,
            "nms_threshold": 0.50
        }

    def get_calibration_config(self) -> Dict[str, Any]:
        return {
            "method": "identity",
            "params": {}
        }

    def get_label_mapping(self) -> Dict[str, str]:
        return {}
