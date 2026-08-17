import time
import torch
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw
from datetime import datetime
from typing import Dict, Any, List, Optional

# GPU memory helper (inline to avoid circular imports)
def _gpu_mem_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1e9
        free, _ = torch.cuda.mem_get_info(0)
        return f"GPU: {alloc:.2f}GB alloc, {free/1e9:.1f}GB free"
    return "GPU: N/A"

class PipelineTracker:
    """
    Tracking and Logging utility for module-level transparency in the pipeline.
    """
    def __init__(self, debug_mode: bool = True, verbose_level: str = "medium", output_dir: str = "debug_outputs"):
        self.debug_mode = debug_mode
        self.verbose_level = verbose_level # "low", "medium", "high"
        self.output_dir = output_dir
        self.stats = []
        self.start_time = time.time()
        self.module_timers = {}
        
        if self.debug_mode:
            os.makedirs(self.output_dir, exist_ok=True)

    def log_module_start(self, module_name: str, input_size: Any = None):
        if self.verbose_level in ["low", "medium", "high"]:
            print(f"\n[MODULE START] {module_name}")
            if input_size is not None:
                print(f"[INPUT SIZE] {input_size}")
            print(f"[{_gpu_mem_str()}]")
        
        self.module_timers[module_name] = time.time()
        return self.module_timers[module_name]

    def log_module_end(self, module_name: str, start_time: float = None, status: str = "EXECUTED", metadata: Dict[str, Any] = None):
        if start_time is None:
            start_time = self.module_timers.get(module_name, time.time())
            
        duration = time.time() - start_time
        
        # Device detection
        device = "CPU"
        if torch.cuda.is_available():
            try:
                found_gpu = False
                if metadata:
                    for v in metadata.values():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            found_gpu = True
                            break
                        if isinstance(v, dict):
                             for vv in v.values():
                                 if isinstance(vv, torch.Tensor) and vv.is_cuda:
                                     found_gpu = True
                                     break
                device = "CUDA" if found_gpu else "CPU"
            except:
                device = "CUDA"
        
        module_stat = {
            "name": module_name,
            "duration": duration,
            "status": status,
            "device": device,
            "metadata": metadata or {}
        }
        self.stats.append(module_stat)

        if self.verbose_level in ["low", "medium", "high"]:
            print(f"[STATUS] {status}")
            print(f"[DEVICE] {device}")
            print(f"[TIME] {duration:.4f} seconds")
            print(f"[{_gpu_mem_str()}]")
            if self.verbose_level in ["medium", "high"]:
                self._print_summary(module_name, metadata, status)

    def log_cached_module(self, module_name: str, metadata: Dict[str, Any] = None):
        """Specifically for when a module is skipped due to cache."""
        if self.verbose_level in ["low", "medium", "high"] and len(self.stats) < 10:
            print(f"\n[MODULE START] {module_name}")
            print(f"[MODULE STATUS] LOADED FROM CACHE")
            print(f"[MODULE END] {module_name} (SKIPPED)")
            if self.verbose_level in ["medium", "high"]:
                self._print_summary(module_name, metadata, "LOADED FROM CACHE")
            
        self.stats.append({
            "name": module_name,
            "duration": 0.0,
            "status": "LOADED FROM CACHE",
            "device": "N/A",
            "metadata": metadata or {}
        })

    def _print_summary(self, module_name: str, metadata: Dict[str, Any], status: str):
        if not metadata:
            return

        print(f"--- {module_name} Summary ({status}) ---")
        
        if "num_boxes" in metadata:
            print(f"  Boxes: {metadata['num_boxes']}")
            if "score_stats" in metadata:
                s = metadata['score_stats']
                print(f"  Scores: Min={s['min']:.3f}, Max={s['max']:.3f}, Mean={s['mean']:.3f}")

        if "num_regions" in metadata:
            print(f"  Regions: {metadata['num_regions']}")
            if "sample_captions" in metadata:
                for i, cap in enumerate(metadata['sample_captions'][:3]):
                    print(f"    - \"{cap[:80]}...\"" if len(cap) > 80 else f"    - \"{cap}\"")
            if "caption_stats" in metadata:
                print(f"  Cap Len: Avg={metadata['caption_stats']['mean_len']:.1f}")

        if "embedding_shape" in metadata:
            print(f"  Embed Shape: {metadata['embedding_shape']}")

        if "dist_stats" in metadata:
            s = metadata['dist_stats']
            print(f"  Distances: Min={s['min']:.3f}, Max={s['max']:.3f}, Mean={s['mean']:.3f}")
            
        if "lpc_stats" in metadata:
            s = metadata['lpc_stats']
            print(f"  LPC: LowDensity={s['low_density_count']}, WeightMean={s['weight_mean']:.3f}")

        if "num_masks" in metadata:
            print(f"  Masks: {metadata['num_masks']}")

        if "score_delta" in metadata:
            s = metadata['score_delta']
            print(f"  SRM Refine: {s['before_mean']:.3f} -> {s['after_mean']:.3f} ({s['change_percent']:+.1f}%)")

        if "pair_count" in metadata:
            print(f"  Pairs: {metadata['pair_count']}")

        print("-" * 40)

    def save_debug_json(self, module_name: str, image_id: str, data: Dict[str, Any]):
        if not self.debug_mode: return
        try:
            import json
            module_dir = os.path.join(self.output_dir, module_name)
            os.makedirs(module_dir, exist_ok=True)
            
            # Convert tensors to lists for JSON serialization
            def serializable(v):
                if isinstance(v, torch.Tensor): return v.detach().cpu().tolist()
                if isinstance(v, np.ndarray): return v.tolist()
                if isinstance(v, dict): return {kk: serializable(vv) for kk, vv in v.items()}
                if isinstance(v, list): return [serializable(vv) for vv in v]
                return v

            clean_data = serializable(data)
            out_path = os.path.join(module_dir, f"{image_id}.json")
            with open(out_path, "w") as f:
                json.dump(clean_data, f, indent=2)
        except Exception as e:
            print(f"[Warning] Failed to save debug JSON: {e}")

    def save_debug_image(self, module_name: str, image_path: str, boxes: torch.Tensor = None, labels: List[str] = None, masks: torch.Tensor = None):
        if not self.debug_mode or not image_path: return
        try:
            module_dir = os.path.join(self.output_dir, module_name)
            os.makedirs(module_dir, exist_ok=True)
            img = Image.open(image_path).convert("RGB")
            if masks is not None:
                img_np = np.array(img)
                if masks.ndim == 2: masks = masks.unsqueeze(0)
                overlay = np.zeros_like(img_np, dtype=np.uint8)
                for i in range(min(10, len(masks))):
                    mask_np = masks[i].cpu().numpy()
                    if mask_np.ndim > 2: mask_np = mask_np.squeeze()
                    mask = mask_np > 0.5
                    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
                img_np = cv2.addWeighted(img_np, 1.0, overlay, 0.4, 0)
                img = Image.fromarray(img_np)
            draw = ImageDraw.Draw(img)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    draw.rectangle(box.tolist(), outline="red", width=3)
            timestamp = datetime.now().strftime("%H%M%S_%f")
            base = os.path.basename(image_path).split('.')[0]
            img.save(os.path.join(module_dir, f"{base}_{timestamp}.png"))
        except: pass

    def print_pipeline_summary(self):
        total_time = time.time() - self.start_time
        module_stats = {}
        for s in self.stats:
            name = s['name']
            if name not in module_stats:
                module_stats[name] = {'count': 0, 'time': 0.0, 'status': {}}
            module_stats[name]['count'] += 1
            module_stats[name]['time'] += s['duration']
            st = s['status']
            module_stats[name]['status'][st] = module_stats[name]['status'].get(st, 0) + 1
            
        print("\n" + "="*85)
        print(f"{'MODULE':<25} | {'COUNT':<6} | {'AVG TIME':<10} | {'TOTAL TIME':<10} | {'STATUS'}")
        print("-" * 85)
        for name, data in module_stats.items():
            avg = data['time'] / data['count']
            st_str = ", ".join([f"{k}:{v}" for k, v in data['status'].items()])
            print(f"{name:<25} | {data['count']:<6} | {avg:8.3f}s | {data['time']:9.2f}s | {st_str}")
        print("-" * 85)
        print(f"PIPELINE TOTAL TIME: {total_time:.2f}s")
        print("="*85 + "\n")

    def log_error(self, module_name: str, error: Exception):
        print(f"\n[ERROR] Module '{module_name}' failed: {error}")
        import traceback; traceback.print_exc()
