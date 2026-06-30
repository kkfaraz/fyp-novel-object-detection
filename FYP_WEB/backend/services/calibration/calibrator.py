import logging
import torch
import numpy as np
import pickle
from typing import Dict, Any

logger = logging.getLogger("Calibrator")

class Calibrator:
    def __init__(self, config: Dict[str, Any]):
        self.method = config.get("method", "identity").lower()
        self.params = config.get("params", {})
        self.pretrained_model = None
        
        if self.method == "isotonic" and "model_path" in self.params:
            try:
                with open(self.params["model_path"], "rb") as f:
                    self.pretrained_model = pickle.load(f)
                logger.info(f"Loaded pretrained Isotonic Regression model from {self.params['model_path']}")
            except Exception as e:
                logger.warning(f"Failed to load Isotonic Regression model: {e}. Falling back to Identity.")
                self.method = "identity"

    def calibrate(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Calibrates detection confidence scores.
        """
        if len(scores) == 0:
            return scores

        if self.method == "identity":
            return scores
            
        elif self.method == "temperature":
            # Temperature scaling for probabilities: p_new = p_old ^ (1 / T)
            # Normalizing/clamping to avoid division by zero
            temp = float(self.params.get("temperature", 1.0))
            if temp <= 0:
                temp = 1.0
            
            # Use power scaling as a temperature scaling approximation on probabilities
            calibrated = torch.pow(scores, 1.0 / temp)
            return torch.clamp(calibrated, 0.0, 1.0)
            
        elif self.method == "platt":
            # Platt Scaling: p_new = 1 / (1 + exp(-(A * p_old + B)))
            a = float(self.params.get("A", 1.0))
            b = float(self.params.get("B", 0.0))
            calibrated = torch.sigmoid(a * scores + b)
            return calibrated
            
        elif self.method == "isotonic":
            if self.pretrained_model is not None:
                scores_np = scores.cpu().numpy()
                try:
                    calibrated_np = self.pretrained_model.predict(scores_np)
                    return torch.tensor(calibrated_np, dtype=scores.dtype, device=scores.device)
                except Exception as e:
                    logger.warning(f"Isotonic prediction failed: {e}")
                    return scores
            return scores
            
        return scores
