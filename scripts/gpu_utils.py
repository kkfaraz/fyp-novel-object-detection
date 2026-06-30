"""
Centralized GPU Device Management Utility
==========================================

Single source of truth for device management across the entire
Novel Object Detection pipeline. Provides:

- Unified device selection
- Runtime device validation
- GPU memory logging
- Tensor/model device checking
- Safe tensor transfer with non-blocking support

Usage:
    from gpu_utils import get_device, log_device_info, validate_model_device
"""

import torch
import gc
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# Core Device Selection
# =============================================================================

def get_device() -> torch.device:
    """
    Get the best available device.
    Returns torch.device("cuda") if GPU is available, else torch.device("cpu").
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_device_info():
    """
    Print comprehensive GPU/device information at pipeline startup.
    Call this once at the beginning of any pipeline run.
    """
    print("\n" + "=" * 70)
    print("GPU DEVICE INFORMATION")
    print("=" * 70)
    print(f"  CUDA Available:     {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA Version:       {torch.version.cuda}")
        print(f"  Device Count:       {torch.cuda.device_count()}")
        print(f"  Current Device:     {torch.cuda.current_device()}")
        print(f"  Device Name:        {torch.cuda.get_device_name(0)}")
        
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem, _ = torch.cuda.mem_get_info(0)
        free_mem_gb = free_mem / 1e9
        
        print(f"  Total GPU Memory:   {total_mem:.2f} GB")
        print(f"  Free GPU Memory:    {free_mem_gb:.2f} GB")
        print(f"  Allocated Memory:   {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Reserved Memory:    {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    else:
        print("  WARNING: No CUDA GPU detected! Pipeline will run on CPU.")
        print("  This will result in significantly slower execution.")
    
    print(f"  PyTorch Version:    {torch.__version__}")
    print(f"  Selected Device:    {get_device()}")
    print("=" * 70 + "\n")


# =============================================================================
# Model Device Validation
# =============================================================================

def validate_model_device(model: torch.nn.Module, model_name: str, expected_device: Optional[torch.device] = None):
    """
    Validate and log the device placement of a model's parameters.
    
    Args:
        model: The PyTorch model to check
        model_name: Human-readable name for logging
        expected_device: If provided, warn if model is not on this device
    """
    try:
        param = next(model.parameters())
        actual_device = param.device
        
        if expected_device is not None and actual_device != expected_device:
            print(f"  [GPU WARNING] {model_name} is on {actual_device}, expected {expected_device}!")
        else:
            print(f"  [GPU ✓] {model_name} → {actual_device}")
        
        return actual_device
    except StopIteration:
        print(f"  [GPU INFO] {model_name} has no parameters (stateless module)")
        return None


def validate_all_models(models_dict: Dict[str, torch.nn.Module], expected_device: torch.device):
    """
    Validate device placement for multiple models at once.
    
    Args:
        models_dict: Dict mapping model names to model objects
        expected_device: Expected device for all models
    """
    print("\n[GPU Validation] Checking model device placement...")
    all_ok = True
    for name, model in models_dict.items():
        if model is None:
            print(f"  [GPU INFO] {name} → None (not loaded)")
            continue
        actual = validate_model_device(model, name, expected_device)
        if actual is not None and actual != expected_device:
            all_ok = False
    
    if all_ok:
        print("[GPU Validation] ✓ All models on correct device\n")
    else:
        print("[GPU Validation] ⚠ Some models on wrong device!\n")
    
    return all_ok


# =============================================================================
# Tensor Device Validation
# =============================================================================

def validate_tensor_device(tensor: torch.Tensor, name: str, expected_device: Optional[torch.device] = None):
    """
    Check and log the device of a tensor. Warn if on CPU when GPU is expected.
    
    Args:
        tensor: The tensor to check
        name: Human-readable name for logging
        expected_device: If provided, warn if tensor is not on this device
    """
    if not isinstance(tensor, torch.Tensor):
        return
    
    actual_device = tensor.device
    if expected_device is not None and actual_device.type != expected_device.type:
        print(f"  [GPU WARNING] Tensor '{name}' is on {actual_device}, expected {expected_device}!")
    
    return actual_device


# =============================================================================
# Safe Tensor Transfer
# =============================================================================

def ensure_tensor_on_device(
    tensor: torch.Tensor, 
    device: torch.device, 
    non_blocking: bool = True
) -> torch.Tensor:
    """
    Safely move a tensor to the target device with non-blocking transfer.
    No-op if tensor is already on the target device.
    
    Args:
        tensor: Tensor to move
        device: Target device
        non_blocking: Use non-blocking transfer (default True for pinned memory)
    
    Returns:
        Tensor on the target device
    """
    if tensor.device == device:
        return tensor
    return tensor.to(device, non_blocking=non_blocking)


def ensure_dict_on_device(d: Dict[str, Any], device: torch.device, non_blocking: bool = True) -> Dict[str, Any]:
    """
    Recursively move all tensors in a dictionary to the target device.
    
    Args:
        d: Dictionary potentially containing tensors
        device: Target device
        non_blocking: Use non-blocking transfer
    
    Returns:
        Dictionary with all tensors moved to device
    """
    result = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            result[k] = ensure_tensor_on_device(v, device, non_blocking)
        elif isinstance(v, dict):
            result[k] = ensure_dict_on_device(v, device, non_blocking)
        elif isinstance(v, list):
            result[k] = [
                ensure_tensor_on_device(item, device, non_blocking) if isinstance(item, torch.Tensor)
                else ensure_dict_on_device(item, device, non_blocking) if isinstance(item, dict)
                else item
                for item in v
            ]
        else:
            result[k] = v
    return result


# =============================================================================
# GPU Memory Logging
# =============================================================================

def log_gpu_memory(stage_name: str = ""):
    """
    Log current GPU memory usage. Call at key pipeline stages.
    
    Args:
        stage_name: Name of the pipeline stage for context
    """
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    free_gb = free_mem / 1e9
    
    prefix = f"[GPU Memory: {stage_name}]" if stage_name else "[GPU Memory]"
    print(f"{prefix} Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | "
          f"Peak: {max_allocated:.2f}GB | Free: {free_gb:.2f}GB")


def clear_gpu_memory():
    """
    Aggressively clear GPU memory cache.
    Call between major pipeline stages or before loading large models.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# =============================================================================
# Pure-Torch MinMaxScaler (GPU-compatible replacement for sklearn)
# =============================================================================

def torch_minmax_scale(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Pure-PyTorch MinMaxScaler implementation. Replaces sklearn.preprocessing.MinMaxScaler
    to avoid CPU round-trips.
    
    Scales tensor values to [0, 1] range along the given dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension along which to compute min/max
    
    Returns:
        Scaled tensor in [0, 1] range, on the same device as input
    """
    if tensor.numel() == 0:
        return tensor
    
    t_min = tensor.min(dim=dim, keepdim=True).values
    t_max = tensor.max(dim=dim, keepdim=True).values
    
    # Avoid division by zero
    denom = t_max - t_min
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    
    return (tensor - t_min) / denom
