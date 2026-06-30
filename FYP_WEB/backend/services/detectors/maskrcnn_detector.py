import logging
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple
from backend.services.detectors.base import BaseDetector
from backend.services.registry.model_registry import ModelRegistry

logger = logging.getLogger("MaskRCNNDetector")

class MaskRCNNDetector(BaseDetector):
    def __init__(self, device: torch.device):
        self.device = device
        self.registry = ModelRegistry()
        self.maskrcnn_d2 = self.registry.load_maskrcnn(device)
        self.maskrcnn_tv = self.registry.load_torchvision_maskrcnn(device)

    def detect(self, inputs: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Base detect method running standard detectron2 supervised forward pass.
        Returns all predictions (boxes, scores, classes).
        """
        self.maskrcnn_d2.eval()
        
        # Configure thresholds
        if not isinstance(self.maskrcnn_d2.proposal_generator, nn.ModuleList):
            self.maskrcnn_d2.proposal_generator.nms_thresh = 0.9
        else:
            self.maskrcnn_d2.proposal_generator.nms_thresh_train = 0.9
            self.maskrcnn_d2.proposal_generator.nms_thresh_test = 0.9

        if isinstance(self.maskrcnn_d2.roi_heads.box_predictor, nn.ModuleList):
            box_predictors = self.maskrcnn_d2.roi_heads.box_predictor
        else:
            box_predictors = [self.maskrcnn_d2.roi_heads.box_predictor]

        for box_predictor in box_predictors:
            box_predictor.allow_novel_classes_during_inference = True
            box_predictor.test_topk_per_image = 300
            box_predictor.test_nms_thresh = 0.5
            box_predictor.test_score_thresh = 0.0001

        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.maskrcnn_d2(inputs)
            
        instances = outputs[0]["instances"]
        return {
            "boxes": instances.pred_boxes.tensor,
            "scores": instances.scores,
            "labels": instances.pred_classes
        }

    def detect_known(self, inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Runs Detectron2 to predict COCO-known classes (labels < 80).
        """
        outputs = self.detect(inputs)
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        classes = outputs["labels"]

        known_mask = classes < 80
        return {
            "boxes": boxes[known_mask],
            "scores": scores[known_mask],
            "labels": classes[known_mask]
        }

    def extract_rpn_proposals(self, file_name: str) -> Dict[str, torch.Tensor]:
        """
        Extracts background proposal bounding boxes using the TorchVision RPN model.
        Scales them back to the original image coordinates.
        """
        # Load and preprocess image
        image = Image.open(file_name).convert("RGB")
        w, h = image.size
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # RPN forward hook capture
        rpn_boxes = []
        rpn_scores = []
        
        def capture_proposals(module, input, output):
            if isinstance(output, tuple) and len(output) >= 2:
                boxes_list, scores_list = output[0], output[1]
                if isinstance(boxes_list, list) and len(boxes_list) > 0:
                    rpn_boxes.append(boxes_list[0])
                    if isinstance(scores_list, list) and len(scores_list) > 0:
                        rpn_scores.append(scores_list[0])
        
        hook = self.maskrcnn_tv.rpn.register_forward_hook(capture_proposals)
        
        # Use torchvision model's device
        model_device = next(self.maskrcnn_tv.parameters()).device
        
        try:
            with torch.no_grad():
                self.maskrcnn_tv.eval()
                _ = self.maskrcnn_tv([image_tensor.to(model_device)])
        finally:
            hook.remove()
        
        if not rpn_boxes or len(rpn_boxes[0]) == 0:
            return {
                "boxes": torch.empty((0, 4), device=self.device),
                "scores": torch.empty((0,), device=self.device)
            }

        boxes = rpn_boxes[0].to(self.device)
        scores = rpn_scores[0].to(self.device) if rpn_scores else torch.ones(len(boxes), device=self.device)

        # Scale proposals back to original scale
        try:
            with torch.no_grad():
                images, _ = self.maskrcnn_tv.transform([image_tensor.to(model_device)])
                resized_h, resized_w = images.image_sizes[0]
                scale_x = w / resized_w
                scale_y = h / resized_h
                
                boxes = boxes.clone()
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
        except Exception as e:
            logger.warning(f"Failed to scale RPN proposals: {e}")

        # Limit to top 500 proposals to optimize memory/speed
        if len(boxes) > 500:
            top_k = 500
            top_indices = torch.topk(scores, min(top_k, len(scores)))[1]
            boxes = boxes[top_indices]
            scores = scores[top_indices]

        return {
            "boxes": boxes,
            "scores": scores
        }
