"""
Utility functions for configuration loading, path resolution, and project-level environment setup.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_project_root() -> Path:
    """
    Get absolute path to the project root directory.

    Returns:
        Path: Absolute path to project root directory.
    """
    # This file is in <project_root>/scripts/config_utils.py
    return Path(__file__).resolve().parent.parent


def resolve_path(path_str: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """
    Resolve a given path string or Path object into an absolute Path relative to project root.

    Args:
        path_str: Path string or Path object (absolute or relative).
        base_dir: Optional base directory to resolve relative paths against. Defaults to project root.

    Returns:
        Path: Resolved absolute path object.
    """
    if base_dir is None:
        base_dir = get_project_root()

    p = Path(path_str)
    
    # If path is already absolute and exists, use it
    if p.is_absolute() and p.exists():
        return p.resolve()

    # If path is absolute but doesn't exist, check if filename/suffix exists relative to project root
    if p.is_absolute():
        rel_candidate = base_dir / p.name
        if rel_candidate.exists():
            return rel_candidate.resolve()

    # Resolve relative path against base_dir
    resolved = (base_dir / p).resolve()
    return resolved


def load_params(params_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a JSON configuration parameter file and resolve machine paths relative to project root.

    Args:
        params_path: Path to params.json configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary with resolved paths.
    """
    proj_root = get_project_root()
    resolved_params_path = resolve_path(params_path, proj_root)

    if not resolved_params_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved_params_path}")

    with open(resolved_params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Resolve standard path keys in params
    path_keys = [
        "detectron2_dir",
        "sam_checkpoint",
        "gdino_checkpoint",
        "cfg_file",
        "rcnn_weight_dir",
        "output_dir",
    ]

    for key in path_keys:
        if key in params and isinstance(params[key], str):
            resolved = resolve_path(params[key], proj_root)
            params[f"{key}_resolved"] = str(resolved)

    return params
