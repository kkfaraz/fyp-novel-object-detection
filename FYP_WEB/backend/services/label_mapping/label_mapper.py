import os
import json
import logging
from pathlib import Path
from backend.services.utils.env_manager import EnvironmentManager

logger = logging.getLogger("LabelMapper")

class LabelMapper:
    def __init__(self):
        root = EnvironmentManager.get_project_root()
        self.mapping_path = root / "backend" / "config" / "label_mapping.json"
        self.mappings = {}
        self._load_mappings()

    def _load_mappings(self):
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, "r") as f:
                    self.mappings = json.load(f)
                logger.info(f"Loaded label mappings from {self.mapping_path}")
            except Exception as e:
                logger.error(f"Failed to load label mapping file: {e}")
                self.mappings = self._get_default_mappings()
        else:
            # Create default mappings
            self.mappings = self._get_default_mappings()
            try:
                # Ensure directory exists
                self.mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.mapping_path, "w") as f:
                    json.dump(self.mappings, f, indent=4)
                logger.info(f"Created default label mapping file at {self.mapping_path}")
            except Exception as e:
                logger.warning(f"Failed to save default label mapping file: {e}")

    def _get_default_mappings(self) -> dict:
        return {
            "tvmonitor": "tv",
            "aeroplane": "airplane",
            "motorbike": "motorcycle",
            "pottedplant": "potted plant",
            "diningtable": "dining table",
            "sofa": "couch"
        }

    def map(self, label: str) -> str:
        """
        Maps a label string to its canonical clean name.
        """
        cleaned = label.strip().replace('_', ' ').lower()
        return self.mappings.get(cleaned, cleaned)
