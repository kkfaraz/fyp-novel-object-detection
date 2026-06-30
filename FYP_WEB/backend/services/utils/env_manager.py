import os
import sys
import torch
from pathlib import Path

class EnvironmentManager:
    _project_root = None

    @staticmethod
    def get_project_root() -> Path:
        if EnvironmentManager._project_root is None:
            # File is located at: <root>/backend/services/utils/env_manager.py
            # So parents[3] is the <root> directory (FYP_WEB)
            EnvironmentManager._project_root = Path(__file__).resolve().parents[3]
        return EnvironmentManager._project_root

    @staticmethod
    def is_colab() -> bool:
        return 'google.colab' in sys.modules

    @staticmethod
    def is_kaggle() -> bool:
        # Kaggle environment check
        return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or any('kaggle' in key.lower() for key in os.environ)

    @staticmethod
    def is_local() -> bool:
        return not (EnvironmentManager.is_colab() or EnvironmentManager.is_kaggle())

    @staticmethod
    def get_env_name() -> str:
        if EnvironmentManager.is_colab():
            return "GOOGLE_COLAB"
        elif EnvironmentManager.is_kaggle():
            return "KAGGLE"
        return "LOCAL"

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @staticmethod
    def get_workers() -> int:
        if EnvironmentManager.is_local():
            return min(4, os.cpu_count() or 1)
        return 1  # Limited resources in Colab/Kaggle

    @staticmethod
    def get_dir_path(dir_name: str) -> Path:
        """
        Get absolute path to a standard directory inside the project root.
        """
        root = EnvironmentManager.get_project_root()
        path = root / "backend" / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path
