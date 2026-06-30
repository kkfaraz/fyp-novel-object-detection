import os
import pickle
import json
import torch
from pathlib import Path

class PipelineOrchestrator:
    def __init__(self, output_dir, exp_name, device="cuda", force_recompute=False):
        self.exp_dir = os.path.join(output_dir, "experiments", exp_name)
        Path(self.exp_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.force_recompute = force_recompute
        
        self.s1_path = os.path.join(self.exp_dir, "stage1_outputs.pkl")
        self.s2_path = os.path.join(self.exp_dir, "stage2_outputs.pkl")
        self.s3_path = os.path.join(self.exp_dir, "stage3_outputs.pkl")
        self.meta_path = os.path.join(self.exp_dir, "metadata.json")

    def save_metadata(self, params):
        print(f"[Metadata] Saving to {self.meta_path}...")
        try:
            with open(self.meta_path, "w") as f:
                json.dump(params, f, indent=4)
            print(f"[Metadata] Saved.")
        except Exception as e:
            print(f"[Metadata] Warning: Could not save metadata: {e}")

    def save_checkpoint(self, data, path):
        print(f"[Checkpoint] Saving to {path}...")
        data_cpu = self.recursive_to_cpu(data)
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "wb") as f:
                pickle.dump(data_cpu, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, path)
            print(f"[Checkpoint] Saved {len(data) if isinstance(data, list) else 'data'}.")
        except Exception as e:
            print(f"[Checkpoint] Failed to save checkpoint to {path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise e

    def load_checkpoint(self, path):
        print(f"[Checkpoint] Loading from {path}...")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError, Exception) as e:
            # Checkpoint file is corrupted – preserve it and signal re-computation
            corrupted_path = path + ".corrupted"
            print(f"[Checkpoint] ⚠ CORRUPTED checkpoint detected: {e}")
            print(f"[Checkpoint] Renaming corrupted file to {corrupted_path}")
            os.rename(path, corrupted_path)
            return None
        if isinstance(data, list):
             print(f"[Checkpoint] Loaded {len(data)} items.")
        else:
             print(f"[Checkpoint] Loaded data.")
        # Move loaded checkpoint data to pipeline device
        if self.device is not None and str(self.device) != "cpu":
            data = self.recursive_to_device(data, self.device)
            print(f"[Checkpoint] Moved data to {self.device}")
        return data

    def should_run(self, stage_path):
        if self.force_recompute:
            return True
        if os.path.exists(stage_path):
            print(f"[Pipeline] Found checkpoint {stage_path}, skipping stage.")
            return False
        return True

    def run_stage(self, stage_id, description, worker_func, *args, **kwargs):
        """
        Generic stage execution wrapper.
        stage_id: 1, 2, or 3 (int)
        description: str for printing
        worker_func: callable that processes data and returns result to be saved
        """
        if stage_id == 1:
            path = self.s1_path
        elif stage_id == 2:
            path = self.s2_path
        elif stage_id == 3:
            path = self.s3_path
        else:
            raise ValueError(f"Invalid stage_id: {stage_id}")

        if not self.should_run(path):
            data = self.load_checkpoint(path)
            if data is not None:
                tracker = kwargs.get("param_dict", {}).get("tracker") if "param_dict" in kwargs else kwargs.get("tracker")
                if tracker:
                    tracker.log_cached_module(f"Stage-{stage_id}-Checkpointed")
                return data
            # Corrupted checkpoint – fall through to re-run the stage
            print(f"[Pipeline] Re-running stage {stage_id} due to corrupted checkpoint.")
            
        print("\n" + "="*60)
        print(f"STAGE {stage_id}: {description}")
        print("="*60)
        
        # Run user provided worker function
        result = worker_func(*args, **kwargs)
        
        self.save_checkpoint(result, path)
        return result

    def recursive_to_cpu(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        elif hasattr(obj, "to") and callable(obj.to):
            return obj.to("cpu")
        elif isinstance(obj, dict):
            return {k: self.recursive_to_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_to_cpu(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.recursive_to_cpu(x) for x in obj)
        return obj

    def recursive_to_device(self, obj, device):
        """Move all tensors in a nested structure to the specified device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif hasattr(obj, "to") and callable(obj.to):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self.recursive_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursive_to_device(x, device) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.recursive_to_device(x, device) for x in obj)
        return obj
