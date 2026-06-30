import logging
from typing import List, Dict, Any, Tuple
import torch
from backend.services.strategies.base import DatasetStrategy

logger = logging.getLogger("DatasetAdapter")

class DatasetAdapter:
    def __init__(self, strategy: DatasetStrategy):
        self.strategy = strategy
        self.dataset_name = strategy.get_dataset_name()
        self.strategy_classes = strategy.get_classes()
        self.coco_classes = [
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
        self.coco_to_strategy = self._build_coco_mapping()

    def _build_coco_mapping(self) -> Dict[int, int]:
        """Builds class index mapping from COCO to target strategy classes using name matching."""
        mapping = {}
        strategy_clean = {c.replace('_', ' ').lower().strip(): idx for idx, c in enumerate(self.strategy_classes)}
        
        # Name overrides for spelling differences (COCO -> Target Strategy)
        overrides = {
            "airplane": ["aeroplane", "airplane"],
            "motorcycle": ["motorbike", "motorcycle"],
            "potted plant": ["pottedplant", "potted plant"],
            "dining table": ["diningtable", "dining table"],
            "tv": ["tvmonitor", "television", "tv"],
            "couch": ["sofa", "couch"]
        }

        for coco_idx, coco_name in enumerate(self.coco_classes):
            coco_name_clean = coco_name.replace('_', ' ').lower().strip()
            
            # Check overrides
            matched = False
            if coco_name_clean in overrides:
                for target_variant in overrides[coco_name_clean]:
                    if target_variant in strategy_clean:
                        mapping[coco_idx] = strategy_clean[target_variant]
                        matched = True
                        break
            
            # Standard match
            if not matched:
                if coco_name_clean in strategy_clean:
                    mapping[coco_idx] = strategy_clean[coco_name_clean]
                else:
                    mapping[coco_idx] = -1  # No mapping found

        return mapping

    def adapt_maskrcnn_known(self, d2_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Maps Mask R-CNN known detections to the strategy classes and standardizes output.
        """
        boxes = d2_outputs["boxes"]
        scores = d2_outputs["scores"]
        labels = d2_outputs["labels"]

        mapped_labels = []
        keep_indices = []

        for idx, lbl in enumerate(labels):
            lbl_item = int(lbl.item())
            mapped_lbl = self.coco_to_strategy.get(lbl_item, -1)
            if mapped_lbl != -1:
                mapped_labels.append(mapped_lbl)
                keep_indices.append(idx)

        if not keep_indices:
            empty_tensor = torch.empty((0,), device=boxes.device)
            return {
                "boxes": torch.empty((0, 4), device=boxes.device),
                "scores": empty_tensor,
                "labels": torch.empty((0,), dtype=torch.long, device=boxes.device)
            }

        keep_indices = torch.tensor(keep_indices, device=boxes.device)
        return {
            "boxes": boxes[keep_indices],
            "scores": scores[keep_indices],
            "labels": torch.tensor(mapped_labels, dtype=torch.long, device=boxes.device)
        }

    def adapt_gdino(self, gdino_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        GDINO outputs are already aligned with strategy classes.
        Just returns them as-is.
        """
        return gdino_outputs

    def adapt_rpn(self, rpn_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        RPN proposals have no labels (labeled as -1 background).
        """
        boxes = rpn_outputs["boxes"]
        scores = rpn_outputs["scores"]
        labels = torch.full((len(boxes),), -1, dtype=torch.long, device=boxes.device)
        
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels
        }
