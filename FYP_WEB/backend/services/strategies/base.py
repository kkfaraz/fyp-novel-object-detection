from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch

class DatasetStrategy(ABC):
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return the lowercase name of the dataset (coco, voc, lvis)"""
        pass

    @abstractmethod
    def get_classes(self) -> List[str]:
        """Return list of class names for this dataset split"""
        pass

    @abstractmethod
    def get_prompts(self, tokenizer, class_len_per_prompt: int = 80) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Build text prompts and positive maps for Grounding DINO.
        """
        pass

    @abstractmethod
    def get_thresholds(self) -> Dict[str, float]:
        """Return confidence, NMS, and other thresholds"""
        pass

    @abstractmethod
    def get_calibration_config(self) -> Dict[str, Any]:
        """Return calibration configuration parameters"""
        pass

    @abstractmethod
    def get_label_mapping(self) -> Dict[str, str]:
        """Return dataset-specific label normalization mappings"""
        pass
