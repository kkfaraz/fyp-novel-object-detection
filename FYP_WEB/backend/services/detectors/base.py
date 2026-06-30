from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray, **kwargs) -> Any:
        """
        Run inference on the image and return raw outputs.
        """
        pass
