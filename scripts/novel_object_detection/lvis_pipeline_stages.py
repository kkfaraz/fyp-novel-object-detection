import torch
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
import os
import gc
import hashlib
import pickle
import time
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ground_dino_utils import run_stage1_inference, batched_mask_to_box
from vlrm_clip_module import Stage2_VLRM_CLIP_Fusion
from srm_module import ScoreRefinementModule
from ot_fusion_module import ot_fusion
from adaptive_proposal_fusion import AdaptiveProposalFusion
from detectron2.structures import Instances, Boxes, pairwise_iou
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.ops import batched_nms
from pipeline_utils import PipelineTracker
from gpu_utils import log_gpu_memory, torch_minmax_scale


def _filter_background_proposals(bg_boxes, bg_objectness, gdino_boxes, iou_threshold=0.5):
    """
    Remove background proposals that overlap with GDINO detections.
    This matches the COCO pipeline's extract_background_proposals() logic,
    ensuring Stage 2 only processes truly novel/unknown regions.

    Args:
        bg_boxes: (N, 4) RPN proposals
        bg_objectness: (N,) objectness scores
        gdino_boxes: (M, 4) GDINO detected boxes
        iou_threshold: Maximum IoU to consider as background (default 0.5)

    Returns:
        filtered_boxes, filtered_objectness
    """
    if len(bg_boxes) == 0 or len(gdino_boxes) == 0:
        return bg_boxes, bg_objectness

    ious = pairwise_iou(Boxes(bg_boxes), Boxes(gdino_boxes))
    max_ious, _ = ious.max(dim=1)
    bg_mask = max_ious < iou_threshold

    filtered_boxes = bg_boxes[bg_mask]
    filtered_objectness = bg_objectness[bg_mask]

    num_removed = len(bg_boxes) - len(filtered_boxes)
    if num_removed > 0:
        print(f"[BG Filter] Removed {num_removed}/{len(bg_boxes)} proposals overlapping GDINO (IoU>{iou_threshold})")

    return filtered_boxes, filtered_objectness

def recursive_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to_cpu(x) for x in obj]
    return obj

def _prepare_detectron2_input(dataset_dict):
    image = cv2.imread(dataset_dict["file_name"])
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {dataset_dict['file_name']}")

    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    prepared = {
        "image": image_tensor,
        "height": dataset_dict.get("height", height),
        "width": dataset_dict.get("width", width),
        "file_name": dataset_dict["file_name"],
        "image_id": dataset_dict.get("image_id", 0),
        "original_dict": dataset_dict
    }
    return prepared

class LVISDataset(Dataset):
    def __init__(self, dataset_dicts):
        self.dataset_dicts = dataset_dicts

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        dataset_dict = self.dataset_dicts[idx]
        return _prepare_detectron2_input(dataset_dict)

def collate_fn(batch):
    return batch

def run_stage1_lvis(dataloader, model=None, text_prompt=None, param_dict=None):
    data_split = param_dict.get("lvis_data_split", param_dict.get("data_split", ""))
    is_voc = "voc" in data_split.lower()
    print(f"[Stage 1] Running GDINO + Mask-RCNN ({'VOC' if is_voc else 'LVIS'})...")
    device = param_dict.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    debug_mode = param_dict.get("debug_mode", True)
    verbose_level = param_dict.get("verbose_level", "medium")
    out_dir = param_dict.get("out_dir", "debug_outputs")
    label_dir = "debug_outputs_voc" if is_voc else "debug_outputs_lvis"
    tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, label_dir))
    param_dict["tracker"] = tracker

    # 5. ENABLE FAST DATALOADER
    # num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2
    num_workers = min(8, os.cpu_count() or 1)
    dataset = LVISDataset(dataloader)
    dataloader_gpu = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn
    )

    cache_dir = "cache/stage1_voc" if is_voc else "cache/stage1_lvis_refined"
    os.makedirs(cache_dir, exist_ok=True)

    results = []
    skipped_count = 0
    saved_count = 0

    for idx, batch in enumerate(tqdm(dataloader_gpu, desc="Stage 1")):
        prepared_input = batch[0]
        image_path = prepared_input["file_name"]
        image_id = prepared_input.get("image_id", 0)

        # Safe filename for image_id
        safe_id = str(image_id).replace('/', '_').replace('\\', '_')
        cache_path = os.path.join(cache_dir, f"{safe_id}.pkl")

        # 2. AUTO-SKIP PROCESSED IMAGES (unless force_recompute)
        force_recompute = param_dict.get("force_recompute", False)
        if os.path.exists(cache_path) and not force_recompute:
            try:
                with open(cache_path, "rb") as f:
                    pkl = pickle.load(f)

                # Reconstruct output dict for downstream stages
                # Map back to: gdino, known, background
                loaded_dict = {
                    "file_name": pkl["meta"]["filename"],
                    "image_id": pkl["image_id"],
                    "height": pkl["meta"]["height"],
                    "width": pkl["meta"]["width"],
                    "known": {
                        "boxes": torch.as_tensor(pkl["known_boxes"]).cpu(),
                        "scores": torch.as_tensor(pkl["known_scores"]).cpu(),
                        "labels": torch.as_tensor(pkl["known_labels"]).cpu()
                    },
                    "background": {
                        "boxes": torch.as_tensor(pkl["background_boxes"]).cpu(),
                        "objectness": torch.as_tensor(pkl["background_objectness"]).cpu()
                    },
                    "gdino": {
                        "boxes": torch.as_tensor(pkl["gdino_boxes"]).cpu(),
                        "scores": torch.as_tensor(pkl["gdino_scores"]).cpu(),
                        "labels": torch.as_tensor(pkl["gdino_labels"]).cpu()
                    }
                }

                results.append(loaded_dict)
                skipped_count += 1
                if skipped_count % 500 == 0 or skipped_count == 1:
                    print(f"[CACHE HIT] image_id={image_id} loaded successfully from cache. [SKIPPED] already processed.")
                continue
            except Exception as e:
                print(f"[CACHE LOAD FAILED] Failed to read {cache_path}: {e}. Deleting corrupted file and recomputing...")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        # 9. ADD FAILURE RECOVERY
        try:
            if device.type == "cuda":
                prepared_input["image"] = prepared_input["image"].to(device, non_blocking=True)

            # 10. OPTIONAL SPEED BOOSTS
            # Use torch.inference_mode() and amp.autocast for maximum GPU speed
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
                out = run_stage1_inference([prepared_input], param_dict)

            # Extract fields to save in exact format specified
            known = out["known"]
            bg = out["background"]
            gdino = out["gdino"]

            pkl_data = {
                "image_id": image_id,
                "gdino_boxes": gdino["boxes"].cpu().numpy(),
                "gdino_scores": gdino["scores"].cpu().numpy(),
                "gdino_labels": gdino["labels"].cpu().numpy(),
                "known_boxes": known["boxes"].cpu().numpy(),
                "known_scores": known["scores"].cpu().numpy(),
                "known_labels": known["labels"].cpu().numpy(),
                "background_boxes": bg["boxes"].cpu().numpy(),
                "background_objectness": bg["objectness"].cpu().numpy(),
                "meta": {
                    "height": out["height"],
                    "width": out["width"],
                    "filename": image_path
                }
            }

            # Write cache file atomically
            temp_path = cache_path + ".tmp"
            try:
                with open(temp_path, "wb") as f:
                    pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(temp_path, cache_path)
                saved_count += 1
                if saved_count % 100 == 0 or saved_count == 1:
                    print(f"[CACHE SAVE] image_id={image_id} processed and saved to {cache_path}")
            except Exception as write_err:
                print(f"[CACHE SAVE WARNING] Failed to save cache {cache_path}: {write_err}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

            results.append(recursive_to_cpu(out))

        except Exception as e:
            print(f"[STAGE 1 FAILURE] Failed processing image_id={image_id} ({image_path}): {e}")
            # Insert dummy output to prevent orchestrator indexing mismatches
            dummy_out = {
                "file_name": image_path,
                "image_id": image_id,
                "height": prepared_input.get("height", 0),
                "width": prepared_input.get("width", 0),
                "known": {
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.int64)
                },
                "background": {
                    "boxes": torch.empty(0, 4),
                    "objectness": torch.empty(0)
                },
                "gdino": {
                    "boxes": torch.empty(0, 4),
                    "scores": torch.empty(0),
                    "labels": torch.empty(0, dtype=torch.int64)
                }
            }
            results.append(dummy_out)

        # 4. BATCH GPU MEMORY CLEANUP
        if (idx + 1) % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[GPU CLEANUP] Batch memory cleanup triggered at image {idx+1}")

    print(f"[STAGE1 COMPLETE] Processed {len(dataloader_gpu)} images ({skipped_count} loaded from cache, {saved_count} newly saved)")
    log_gpu_memory("after_stage1_lvis")
    return results

def _batch_load_stage2_from_cache(stage1_results, cache_dir, device):
    """
    Batch-load all Stage-2 results from per-image cache files.
    Skips VLRM/CLIP model initialization entirely.
    Uses a single os.listdir() for O(1) set lookups.
    
    Returns:
        list of stage2 output dicts if enhanced cache files found,
        None if ALL cache files are legacy format (signals caller to fall back).
    """
    t0 = time.time()
    cache_files = set(os.listdir(cache_dir))
    results = []
    loaded = 0
    missing = 0
    legacy = 0

    for s1_out in tqdm(stage1_results, desc="Stage 2 (batch cache load)"):
        if s1_out is None:
            results.append(None)
            continue

        image_id = str(s1_out.get("image_id", os.path.basename(s1_out["file_name"]).split('.')[0]))
        image_path = s1_out["file_name"]
        cache_name = f"{image_id}.pkl"

        if cache_name in cache_files:
            cache_path = os.path.join(cache_dir, cache_name)
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)

                # Use pre-computed stage2 outputs if available (enhanced cache)
                if "stage2_boxes" in cache_data and cache_data["stage2_boxes"] is not None:
                    stage2_out = {
                        "stage2_boxes": torch.as_tensor(cache_data["stage2_boxes"]).cpu(),
                        "stage2_scores": torch.as_tensor(cache_data["stage2_scores"]).cpu(),
                        "stage2_classes": torch.as_tensor(cache_data["stage2_classes"]).cpu(),
                        "file_name": image_path,
                        "image_id": image_id,
                        "height": s1_out.get("height"),
                        "width": s1_out.get("width")
                    }
                    loaded += 1
                else:
                    # Legacy cache — count but don't emit empty tensors yet
                    stage2_out = None  # placeholder
                    legacy += 1

                results.append(stage2_out)
            except Exception as e:
                print(f"[Stage 2 Cache] Failed to load {cache_name}: {e}. Deleting corrupted file.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
                results.append(None)
                missing += 1
        else:
            results.append(None)
            missing += 1

    elapsed = time.time() - t0
    print(f"[Stage 2] Batch loaded {loaded}/{len(stage1_results)} from cache in {elapsed:.1f}s")

    # ── CRITICAL: If cache is incomplete, signal fallback ──
    # This prevents silently returning empty Stage-2 outputs for missing/legacy images.
    if loaded < len(stage1_results):
        print(f"[Stage 2] ⚠ Incomplete cache: loaded {loaded}/{len(stage1_results)} from cache.")
        print(f"[Stage 2] ⚠ Falling back to full VLRM+CLIP pipeline to backfill/generate cache...")
        return None  # Signal caller to run full pipeline

    # Fill in None entries (legacy/missing) with empty tensors
    for i in range(len(results)):
        if results[i] is None and stage1_results[i] is not None:
            s1_ref = stage1_results[i]
            results[i] = {
                "stage2_boxes": torch.empty(0, 4),
                "stage2_scores": torch.empty(0),
                "stage2_classes": torch.empty(0, dtype=torch.int64),
                "file_name": s1_ref["file_name"],
                "image_id": s1_ref.get("image_id", os.path.basename(s1_ref["file_name"]).split('.')[0]),
                "height": s1_ref.get("height"),
                "width": s1_ref.get("width")
            }

    return results


def _save_enhanced_cache(cache_path, cache_data, labeled_boxes, labeled_scores, labeled_classes):
    """
    Update a VLRM cache file to include pre-computed stage2 outputs.
    This enables the fast batch-load path to skip CLIP re-encoding.
    """
    temp_path = cache_path + ".tmp"
    try:
        cache_data["stage2_boxes"] = labeled_boxes.cpu().numpy() if isinstance(labeled_boxes, torch.Tensor) else labeled_boxes
        cache_data["stage2_scores"] = labeled_scores.cpu().numpy() if isinstance(labeled_scores, torch.Tensor) else labeled_scores
        cache_data["stage2_classes"] = labeled_classes.cpu().numpy() if isinstance(labeled_classes, torch.Tensor) else labeled_classes
        with open(temp_path, "wb") as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_path, cache_path)
    except Exception as e:
        print(f"[Stage 2 Cache] Failed to save enhanced cache {os.path.basename(cache_path)}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def run_stage2_lvis(stage1_results, param_dict=None):
    print("[LVIS Stage 2] Running VLRM + CLIP on background proposals...")

    # Clear GPU memory from Stage 1 before initializing Stage 2 models
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vlrm_model_name = param_dict.get("vlrm_model_name", "sashakunitsyn/vlrm-blip2-opt-2.7b")
    clip_model_name = param_dict.get("clip_model_name", "ViT-SO400M-14-SigLIP")
    clip_pretrained = param_dict.get("clip_pretrained", "webli")
    lvis_classes = param_dict.get("lvis_classes", [])
    device = param_dict.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    force_recompute = param_dict.get("force_recompute", False)

    data_split = param_dict.get("lvis_data_split", param_dict.get("data_split", ""))
    is_voc = "voc" in data_split.lower()

    debug_mode = param_dict.get("debug_mode", True)
    verbose_level = param_dict.get("verbose_level", "medium")
    out_dir = param_dict.get("out_dir", "debug_outputs")
    tracker = param_dict.get("tracker")
    if tracker is None:
        label_dir = "debug_outputs_voc" if is_voc else "debug_outputs_lvis"
        tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, label_dir))

    lvis_cache_dir = param_dict.get("vlrm_cache_dir", "cache/vlrm_outputs_lvis")
    os.makedirs(lvis_cache_dir, exist_ok=True)

    print(f"[Stage 2] VLRM cache directory: {lvis_cache_dir}")

    # ── INDEX-BASED COMPLETION CHECK ──────────────────────────────────────────
    # Try the fast batch-load path first: if ALL images have enhanced cache
    # (stage2_boxes saved), skip VLRM/CLIP model initialization entirely.
    # Falls back to full pipeline if any cache files are legacy format.
    # ─────────────────────────────────────────────────────────────────────────
    image_ids = []
    for s1 in stage1_results:
        if s1 is not None:
            image_ids.append(str(s1.get("image_id", os.path.basename(s1["file_name"]).split('.')[0])))

    # Re-enable fast path: skip redundant CLIP re-encoding when enhanced cache exists
    # recompute_stage2_scores=True forces Mahalanobis re-scoring (needed when config changes
    # like SAEG, gamma/temperature removal, fusion_alpha, etc.)
    recompute_scores = param_dict.get("recompute_stage2_scores", False)
    use_fast_path = not force_recompute and not recompute_scores

    # ── FAST PATH: Try batch-loading pre-computed stage2 outputs ───────────────
    if use_fast_path:
        print(f"[Stage 2] Attempting fast batch-load from enhanced cache...")
        batch_results = _batch_load_stage2_from_cache(stage1_results, lvis_cache_dir, device)
        if batch_results is not None:
            print(f"[Stage 2] ✓ Fast path succeeded — skipped VLRM/CLIP model loading entirely")
            log_gpu_memory("after_stage2_lvis_fast")
            return batch_results
        print(f"[Stage 2] Fast path failed — falling back to full pipeline")

    # ── PARTIAL RUN: Pre-build cache index for O(1) lookups ───────────────────
    existing_cache = set()
    if os.path.isdir(lvis_cache_dir):
        existing_cache = set(os.listdir(lvis_cache_dir))
    print(f"[Stage 2] Pre-built cache index: {len(existing_cache)} existing files (O(1) lookups)")

    # ── DEFERRED MODEL INIT: Only load VLRM/CLIP if we actually need to generate ──
    # fusion_alpha=0.55: LVIS rare classes benefit from higher caption weight
    # since VLRM captions are more descriptive for rare objects than CLIP image features.
    fusion_alpha = param_dict.get("fusion_alpha", 0.55)
    print(f"[Stage 2] Using fusion_alpha={fusion_alpha}")

    use_saeg = param_dict.get("use_saeg", False)
    stage2_module = Stage2_VLRM_CLIP_Fusion(
        device=device,
        fusion_alpha=fusion_alpha,
        vlrm_model_name=vlrm_model_name,
        clip_model_name=clip_model_name,
        clip_pretrained=clip_pretrained,
        cache_dir=lvis_cache_dir,
        use_lpc=True,
        use_saeg=use_saeg,
    )

    # ── RAW FEATURES CHECKPOINT (saves 1.5h on resume) ──────────────────────
    raw_checkpoint_dir = os.path.join(os.path.dirname(lvis_cache_dir), "stage2_checkpoints")
    os.makedirs(raw_checkpoint_dir, exist_ok=True)
    total_expected = len(stage1_results)
    raw_checkpoint_path = os.path.join(raw_checkpoint_dir, f"raw_features_{total_expected}.pkl")

    raw_features_list = None
    if not force_recompute and os.path.exists(raw_checkpoint_path):
        try:
            with open(raw_checkpoint_path, "rb") as f:
                raw_features_list = pickle.load(f)
            generated = sum(1 for r in raw_features_list if r is not None and not r.get("empty", True))
            print(f"[Stage 2] ✓ Loaded raw features checkpoint ({len(raw_features_list)} images, {generated} generated). Skipping feature extraction.")
        except Exception as e:
            print(f"[Stage 2] Corrupted checkpoint: {e}. Re-extracting features.")
            raw_features_list = None

    results = []
    stage2_threshold = param_dict.get("stage2_threshold", 0.10)

    # Dynamic calibration for LVIS 1203-class softmax sparsity
    lvis_split = param_dict.get("lvis_data_split", param_dict.get("data_split", "lvis_v1_val"))
    is_voc_check = "voc" in lvis_split.lower()
    if not is_voc_check and stage2_threshold == 0.10:
        stage2_threshold = 0.02
        print(f"[Stage 2] LVIS detected: dynamically adjusting stage2_threshold to {stage2_threshold} to compensate for 1203-class softmax sparsity.")

    if raw_features_list is not None:
        generated = sum(1 for r in raw_features_list if r is not None and not r.get("empty", True))
    else:
        raw_features_list = []
        generated = 0
        dataset_name = "VOC" if "voc" in param_dict.get("lvis_data_split", "").lower() else "LVIS"
        print(f"[Stage 2] Using threshold: {stage2_threshold} for {dataset_name} ({len(lvis_classes)} classes)")

        for idx, s1_out in enumerate(tqdm(stage1_results, desc="Stage 2 Features")):
            if s1_out is None:
                raw_features_list.append(None)
                continue

            bg_boxes = s1_out["background"]["boxes"]
            bg_objectness = s1_out["background"]["objectness"]
            gdino_boxes = s1_out["gdino"]["boxes"]
            image_path = s1_out["file_name"]
            image_id = str(s1_out.get("image_id", os.path.basename(image_path).split('.')[0]))

            bg_boxes, bg_objectness = _filter_background_proposals(
                bg_boxes, bg_objectness, gdino_boxes, iou_threshold=0.5
            )

            if len(bg_boxes) == 0:
                raw_features_list.append({
                    "empty": True,
                    "image_path": image_path,
                    "image_id": s1_out.get("image_id", image_id),
                    "height": s1_out.get("height"),
                    "width": s1_out.get("width"),
                    "cache_path": None
                })
                continue

            cache_name = f"{image_id}.pkl"
            cache_path = os.path.join(lvis_cache_dir, cache_name)

            try:
                full_image = Image.open(image_path).convert("RGB")

                caption_features, image_features, valid_indices = stage2_module.run_stage_2(
                    background_rois=bg_boxes.to(device),
                    full_image=full_image,
                    class_names=lvis_classes,
                    image_id=image_id,
                    image_path=image_path,
                    max_regions=150,
                    verbose=False,
                    return_raw_features=True
                )

                raw_features_list.append({
                    "empty": False,
                    "caption_features": caption_features.cpu(),
                    "image_features": image_features.cpu(),
                    "valid_indices": valid_indices,
                    "bg_boxes": bg_boxes.cpu(),
                    "image_path": image_path,
                    "image_id": s1_out.get("image_id", image_id),
                    "height": s1_out.get("height"),
                    "width": s1_out.get("width"),
                    "cache_path": cache_path
                })
                generated += 1

            except Exception as e:
                print(f"[Stage 2] Error processing {image_path}: {e}")
                raw_features_list.append({
                    "empty": True,
                    "image_path": image_path,
                    "image_id": s1_out.get("image_id", image_id),
                    "height": s1_out.get("height"),
                    "width": s1_out.get("width"),
                    "cache_path": None
                })

        print(f"[Stage 2] Feature extraction complete. Total generated: {generated}")

        # Offload VLRM model to CPU after Stage 2 feature extraction loop completes
        if hasattr(stage2_module, "vlrm"):
            print("[VRAM] Offloading VLRM model to CPU after Stage 2 feature extraction...")
            stage2_module.vlrm.finish(force_offload=True)


        # Save raw features checkpoint so we never lose the 1.5h extraction
        try:
            with open(raw_checkpoint_path, "wb") as f:
                pickle.dump(raw_features_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[Stage 2] ✓ Saved raw features checkpoint ({len(raw_features_list)} images) to {raw_checkpoint_path}")
        except Exception as e:
            print(f"[Stage 2] Warning: Could not save raw features checkpoint: {e}")

    print("[Stage 2] Now calculating Mahalanobis distances and applying thresholding at the end of Stage 2...")

    # Pre-compute the Mahalanobis template statistics for the 1203 LVIS classes on GPU
    mean_features, cov_inv = stage2_module._get_class_stats(lvis_classes)

    for item in tqdm(raw_features_list, desc="Stage 2 Mahalanobis"):
        if item is None:
            results.append(None)
            continue

        if item["empty"]:
            stage2_out = {
                "stage2_boxes": torch.empty(0, 4),
                "stage2_scores": torch.empty(0),
                "stage2_classes": torch.empty(0, dtype=torch.long),
                "file_name": item["image_path"],
                "image_id": item["image_id"],
                "height": item["height"],
                "width": item["width"]
            }
            results.append(stage2_out)
            continue

        caption_features = item["caption_features"].to(device)
        image_features = item["image_features"].to(device)
        valid_indices = item["valid_indices"]
        bg_boxes = item["bg_boxes"].to(device)

        caption_scores_raw = stage2_module.clip.compute_mahalanobis(caption_features, mean_features, cov_inv)
        image_scores_raw = stage2_module.clip.compute_mahalanobis(image_features, mean_features, cov_inv)

        # sigmoid gives per-class independent scores (like baseline CLIP approach),
        # avoiding the extreme sparsity of softmax over 1203 classes.
        caption_sim = torch.sigmoid(caption_scores_raw)
        image_sim = torch.sigmoid(image_scores_raw)

        # Handle edge case where image_features is empty but captions exist
        if caption_sim.shape[0] > 0 and image_sim.shape[0] == 0:
            semantic_scores = caption_sim
        elif image_sim.shape[0] > 0 and caption_sim.shape[0] == 0:
            semantic_scores = image_sim
        elif caption_sim.shape[0] == 0 and image_sim.shape[0] == 0:
            # K must come from mean_features (number of LVIS classes), not an undefined variable
            K = mean_features.shape[0]
            semantic_scores = torch.empty(0, K, device=device)
        else:
            semantic_scores = stage2_module.fusion_alpha * caption_sim + (1 - stage2_module.fusion_alpha) * image_sim

        # LPC if enabled
        if stage2_module.use_lpc:
            semantic_scores = stage2_module._apply_lpc_logic(semantic_scores, image_features, mean_features, item["image_id"], None)

        labeled_boxes = torch.empty(0, 4, device=device)
        labeled_scores = torch.empty(0, device=device)
        labeled_classes = torch.empty(0, dtype=torch.long, device=device)

        if semantic_scores.numel() > 0 and len(valid_indices) > 0:
            conf, cls_idx = torch.max(semantic_scores, dim=1)
            mask = conf > stage2_threshold

            keep_local = torch.nonzero(mask).squeeze(1)
            # Safeguard against list index out of range if shapes mismatch
            keep_local = keep_local[keep_local < len(valid_indices)]
            if keep_local.numel() > 0:
                orig_indices = [valid_indices[k] for k in keep_local.tolist()]
                labeled_boxes = bg_boxes[orig_indices]
                labeled_scores = conf[keep_local]
                labeled_classes = cls_idx[keep_local]

        # Save enhanced cache with stage2 final outputs
        cache_path = item["cache_path"]
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                _save_enhanced_cache(cache_path, cache_data, labeled_boxes, labeled_scores, labeled_classes)
            except Exception as e:
                print(f"[Stage 2 Cache] Failed to load {cache_path} for enhancement: {e}. Deleting corrupted file.")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        stage2_out = {
            "stage2_boxes": labeled_boxes,
            "stage2_scores": labeled_scores,
            "stage2_classes": labeled_classes,
            "file_name": item["image_path"],
            "image_id": item["image_id"],
            "height": item["height"],
            "width": item["width"]
        }
        results.append(recursive_to_cpu(stage2_out))

    print(f"[Stage 2] Complete: processed {len(results)} images.")
    del stage2_module
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("after_stage2_lvis")
    return results


def run_stage3_lvis(stage1_results, stage2_results, param_dict, evaluator):
    data_split = param_dict.get("lvis_data_split", param_dict.get("data_split", ""))
    is_voc = "voc" in data_split.lower()
    print(f"[Stage 3] Merging and Evaluating ({'VOC' if is_voc else 'LVIS'})...")

    sam = param_dict.get("sam")
    debug_mode = param_dict.get("debug_mode", True)
    device = param_dict.get("device", "cuda")

    tracker = param_dict.get("tracker")
    if tracker is None:
        verbose_level = param_dict.get("verbose_level", "medium")
        out_dir = param_dict.get("out_dir", "debug_outputs")
        label_dir = "debug_outputs_voc" if is_voc else "debug_outputs_lvis"
        tracker = PipelineTracker(debug_mode=debug_mode, verbose_level=verbose_level, output_dir=os.path.join(out_dir, label_dir))
        param_dict["tracker"] = tracker

    use_srm = param_dict.get("use_srm", True)
    srm = ScoreRefinementModule(per_image_norm=False) if use_srm else None
    max_before_sam = param_dict.get("max_before_sam", 500)
    max_dets_per_image = param_dict.get("max_dets_per_image", 500)
    if use_srm:
        print("[SRM] Enabled for LVIS Stage 3")

    # Adaptive Proposal Fusion (Disable for VOC to use standard NMS)
    is_voc = "voc" in data_split.lower()
    use_apf = param_dict.get("use_apf", False) if not is_voc else False
    if use_apf:
        apf = AdaptiveProposalFusion(
            source_weights=param_dict.get("apf_source_weights", {0: 0.55, 1: 0.50, 2: 0.40}),
            wbf_iou_threshold=param_dict.get("apf_iou_threshold", 0.5),
            wbf_score_threshold=param_dict.get("apf_score_threshold", 0.05),
            wbf_max_boxes=param_dict.get("apf_max_boxes", 500),
            device=device,
        )
        print(f"[APF] Enabled — WBF(iou={apf.wbf_iou_threshold}, max={apf.wbf_max_boxes})")
    else:
        apf = None

    # Initialize LVISCalibrator — multiplicative boosts compensate for the
    # natural suppression of rare classes in the 1203-class softmax.
    # Stage 2 gamma scaling (Mod 5) adjusts Mahalanobis distances, but the
    # final detection scores still need frequency-aware calibration.
    # Logit adjustment (additive, tau=0.3) provides theoretically grounded
    # rare-class correction without over-boosting.
    from lvis_calibration import LVISCalibrator
    lvis_split = param_dict.get("lvis_data_split", param_dict.get("data_split", "lvis_v1_val"))
    rare_boost = param_dict.get("rare_boost", 1.15)
    common_boost = param_dict.get("common_boost", 1.05)
    frequent_boost = param_dict.get("frequent_boost", 1.0)
    tau = param_dict.get("tau", 0.1)

    if "voc" in lvis_split.lower():
        calibrator = None
    else:
        calibrator = LVISCalibrator(
            data_split=lvis_split,
            num_classes=1203,
            rare_boost=rare_boost,
            common_boost=common_boost,
            frequent_boost=frequent_boost,
            device=device
        )

    resize_transform = param_dict.get("resize_transform")
    verbose = param_dict.get("verbose", False)

    if len(stage1_results) != len(stage2_results):
        print(f"Error: Mismatch S1 vs S2 results.")
        return {}

    if evaluator is not None:
        evaluator.reset()

    # Always use sam_cache — SAM3 has been replaced with standard SAM (ViT-L).
    # Using a fixed name ensures no stale sam3_cache entries are accidentally loaded.
    sam_cache_dir = os.path.join(param_dict.get("out_dir", "outputs_lvis"), "sam_cache")

    existing_cached = []
    if sam_cache_dir and os.path.exists(sam_cache_dir):
        existing_cached = [f for f in os.listdir(sam_cache_dir) if f.startswith("final_")]
        print(f"[Stage 3] Found {len(existing_cached)} cached results. Resuming with robust validation.")

    final_results = []
    processed_count = 0
    skipped_count = 0

    for i in tqdm(range(len(stage1_results)), desc="Stage 3"):
        s1 = stage1_results[i]
        s2 = stage2_results[i]

        if s1 is None:
            continue

        fname = s1["file_name"]
        height = s1["height"]
        width = s1["width"]
        image_id = s1.get("image_id")

        # Compute current config and Stage 2 hash to validate cache validity
        def _to_bytes(t):
            if isinstance(t, torch.Tensor):
                return t.cpu().numpy().tobytes()
            elif hasattr(t, "tobytes"):
                return t.tobytes()
            return str(t).encode('utf-8')
        s2_data = _to_bytes(s2["stage2_boxes"]) + _to_bytes(s2["stage2_scores"]) + _to_bytes(s2["stage2_classes"])
        s2_hash = hashlib.md5(s2_data).hexdigest()[:12]

        current_config = {
            "s2_hash": s2_hash,
            "use_srm": use_srm,
            "max_before_sam": max_before_sam,
            "max_dets_per_image": max_dets_per_image,
            "is_voc": is_voc,
            "sam_checkpoint": param_dict.get("sam_checkpoint"),
            "rare_boost": rare_boost,
            "common_boost": common_boost,
            "frequent_boost": frequent_boost,
            "tau": tau,
            "remove_minmax_bug": True,
            "skip_known_sam_srm": True,
            "ot_penalty_factor": 0.7,
            "pre_sam_nms_iou": 0.65,
        }

        cached_result_path = None
        if sam_cache_dir:
            safe_id = str(image_id).replace('/', '_').replace('\\', '_') if image_id else os.path.basename(fname).split('.')[0]
            cached_result_path = os.path.join(sam_cache_dir, f"final_{safe_id}.pkl")

        if cached_result_path and os.path.exists(cached_result_path):
            try:
                cached = torch.load(cached_result_path, map_location='cpu')
                cached_config = cached.get("config", {})
                
                configs_match = True
                for k, v in current_config.items():
                    if cached_config.get(k) != v:
                        configs_match = False
                        break
                
                if configs_match:
                    inst = Instances((height, width))
                    inst.pred_boxes = Boxes(cached['boxes'])
                    inst.scores = cached['scores']
                    inst.pred_classes = cached['classes']
                    
                    stage3_item = {
                        "file_name": fname,
                        "image_id": image_id,
                        "height": height,
                        "width": width,
                        "instances": inst
                    }
                    final_results.append(stage3_item)
                    
                    if evaluator is not None:
                        inputs_mock = [{
                            "file_name": fname,
                            "height": height,
                            "width": width,
                            "image_id": image_id
                        }]
                        evaluator.process(inputs_mock, [{"instances": inst}])
                    skipped_count += 1
                    continue
            except Exception as e:
                print(f"[Resume] Failed to load cached result for {image_id}: {e}")

        g_boxes = s1["gdino"]["boxes"].to(device)
        g_scores = s1["gdino"]["scores"].to(device)
        g_labels = s1["gdino"]["labels"].to(device)

        known_boxes = s1["known"]["boxes"].to(device)
        known_scores = s1["known"]["scores"].to(device)
        known_labels = s1["known"]["labels"].to(device)

        stage2_thresh = param_dict.get("stage2_threshold", 0.10)
        s2_boxes = s2["stage2_boxes"].to(device) if len(s2["stage2_boxes"]) > 0 else torch.empty(0, 4, device=device)
        s2_scores = s2["stage2_scores"].to(device) if len(s2["stage2_scores"]) > 0 else torch.empty(0, device=device)
        s2_classes = s2["stage2_classes"].to(device) if len(s2["stage2_classes"]) > 0 else torch.empty(0, dtype=torch.int64, device=device)

        # NOTE: Stage 2 already filters by stage2_threshold before saving.
        # Applying a second threshold here would incorrectly re-filter detections
        # that were already kept by Stage 2. Removed to prevent double-filtering.
        # (If you need to test different thresholds, clear Stage 2 cache and rerun.)

        t_start_ot = tracker.log_module_start("Hungarian-OT-Fusion") if tracker else None
        if is_voc:
            # VOC mode: All classes are in the same class space.
            # But known detections from Mask-RCNN are high quality. Keeping them separate
            # during OT matching avoids the unmatched penalty suppressing them.
            # Match Stage 2 detections against GDINO.
            merged_boxes, merged_scores, merged_classes = ot_fusion(
                s2_boxes, s2_scores, s2_classes,
                known_boxes, known_scores, known_labels,
                g_boxes, g_scores, g_labels,
                iou_weight=0.6, semantic_weight=0.4, sinkhorn_reg=0.1, sinkhorn_iters=50,
                boost_factor=1.5, penalty_factor=0.7,
                param_dict={'verbose': verbose, 'img_h': height, 'img_w': width}
            )
        else:
            merged_boxes, merged_scores, merged_classes = ot_fusion(
                s2_boxes, s2_scores, s2_classes,
                known_boxes, known_scores, known_labels,
                g_boxes, g_scores, g_labels,
                iou_weight=0.6, semantic_weight=0.4, sinkhorn_reg=0.1, sinkhorn_iters=50,
                boost_factor=1.5, penalty_factor=0.7,
                param_dict={'verbose': verbose, 'img_h': height, 'img_w': width}
            )
            
        if tracker:
            tracker.log_module_end("Hungarian-OT-Fusion", t_start_ot, metadata={
                "pair_count": len(merged_boxes),
                "cost_summary": {"total_cost": 0}
            })

        # Mask-RCNN detections are already high quality and refined.
        # Construct an already_refined boolean mask to skip SAM and SRM for them,
        # preventing score degradation and preserving the known classes' AP at 55+.
        num_known = len(known_boxes)
        already_refined = torch.zeros(len(merged_boxes), dtype=torch.bool, device=device)
        if num_known > 0:
            already_refined[:num_known] = True

        # Apply SAM, SRM, NMS to all boxes
        final_boxes, final_scores, final_classes = _apply_sam_srm_nms(
            merged_boxes, merged_scores, merged_classes,
            fname, sam, srm, resize_transform, verbose, device, tracker=tracker,
            debug_mode=debug_mode, sam_cache_dir=sam_cache_dir, calibrator=calibrator,
            already_refined=already_refined,
            max_before_sam=max_before_sam, max_dets_per_image=max_dets_per_image,
            is_voc=is_voc, tau=tau
        )

        if tracker and debug_mode and len(final_boxes) > 0:
            tracker.save_debug_image("Final-Detection", fname, boxes=final_boxes[:10])

        inst = Instances((height, width))
        inst.pred_boxes = Boxes(final_boxes)
        inst.scores = final_scores
        inst.pred_classes = final_classes

        result = [{"instances": inst}]
        stage3_item = {
            "file_name": fname,
            "image_id": image_id,
            "height": height,
            "width": width,
            "instances": inst
        }
        final_results.append(stage3_item)

        if cached_result_path:
            temp_path = cached_result_path + ".tmp"
            try:
                os.makedirs(sam_cache_dir, exist_ok=True)
                torch.save({
                    'boxes': final_boxes.cpu(),
                    'scores': final_scores.cpu(),
                    'classes': final_classes.cpu(),
                    'image_id': image_id,
                    'config': current_config
                }, temp_path)
                os.replace(temp_path, cached_result_path)
            except Exception as e:
                print(f"[Resume] Failed to save result: {e}")
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

        if evaluator is not None:
            inputs_mock = [{
                "file_name": fname,
                "height": height,
                "width": width,
                "image_id": image_id
            }]
            evaluator.process(inputs_mock, result)

    if calibrator is not None:
        calibrator.print_stats()

    if evaluator is not None:
        results = evaluator.evaluate()
        print(f"[Resume] Stage 3 complete: {skipped_count} skipped (cached), {len(stage1_results)-skipped_count} processed")
        return results

    print(f"[Resume] Stage 3 complete: {skipped_count} skipped (cached), {len(stage1_results)-skipped_count} processed")
    return final_results


def _get_sam_cache_path(image_path, image_id, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    safe_id = str(image_id).replace('/', '_').replace('\\', '_')
    return os.path.join(cache_dir, f"sam_{safe_id}.pkl")



def _apply_sam_srm_nms(boxes, scores, classes, image_path, sam, srm, resize_transform, verbose, device, tracker=None, debug_mode=False, sam_cache_dir=None, calibrator=None, apf=None, already_refined=None, max_before_sam=500, max_dets_per_image=500, is_voc=False, tau=0.1):
    if len(boxes) == 0:
        return boxes, scores, classes

    # Ensure correct device and float types
    boxes = boxes.float().to(device)
    scores = scores.float().to(device)
    classes = classes.long().to(device)

    if already_refined is None:
        already_refined = torch.zeros(len(boxes), dtype=torch.bool, device=device)
    else:
        already_refined = already_refined.to(device)

    # 1. Pre-SAM deduplication — iou=0.65 for LVIS.
    if len(boxes) > 0:
        keep = batched_nms(boxes, scores, classes, iou_threshold=0.65)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        already_refined = already_refined[keep]

    # 2. Pre-SAM Top-K cap (Change 6)
    if len(scores) > max_before_sam:
        topk_scores, topk_idxs = torch.topk(scores, max_before_sam)
        boxes = boxes[topk_idxs]
        scores = topk_scores
        classes = classes[topk_idxs]
        already_refined = already_refined[topk_idxs]

    sam_scores = None
    sam_refined = None
    image_id = os.path.basename(image_path).split('.')[0]

    cache_available = sam_cache_dir is not None
    cache_path = _get_sam_cache_path(image_path, image_id, sam_cache_dir) if cache_available else None

    # Content hash of input boxes for cache validation (Fix #4)
    # This detects stale cache even when box count matches but content changed.
    boxes_hash = hashlib.md5(boxes.cpu().numpy().tobytes()).hexdigest()[:12]

    if cache_available and cache_path and os.path.exists(cache_path):
        try:
            cached = torch.load(cache_path, map_location=device)
            cached_sam_scores = cached['sam_scores'].to(device)
            cached_sam_refined = cached['sam_refined'].to(device)
            cached_hash = cached.get('boxes_hash', None)

            # Validate both shape AND content hash.
            # If Stage-2 output changed (e.g., was empty before, now has detections),
            # the cached SAM data is stale and will cause a size mismatch in NMS.
            if len(cached_sam_refined) == len(scores) and cached_hash == boxes_hash:
                sam_scores = cached_sam_scores
                sam_refined = cached_sam_refined
                # Do NOT overwrite boxes yet, we will select better box later.
                print(f"[SAM Cache] Loaded: {os.path.basename(cache_path)}")
            else:
                reason = "count mismatch" if len(cached_sam_refined) != len(scores) else "content changed"
                print(f"[SAM Cache] STALE ({reason}): {os.path.basename(cache_path)} — re-running SAM")
                # Delete stale cache file so it gets regenerated
                try:
                    os.remove(cache_path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[SAM Cache] Load failed for {image_id}: {e}. Deleting corrupted file.")
            try:
                os.remove(cache_path)
            except OSError:
                pass

    if sam is not None and sam_refined is None:
        try:
            if resize_transform is None:
                resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

            curr_image = cv2.imread(image_path)
            curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            img_shape = curr_image.shape[:2]

            curr_image = resize_transform.apply_image(curr_image)
            curr_image = torch.as_tensor(curr_image, device=sam.device).permute(2, 0, 1).contiguous()

            # Initialize outputs with default values (so already_refined boxes remain unchanged)
            sam_refined = boxes.clone()
            sam_scores = torch.ones_like(scores) # default IoU prediction = 1.0 for already refined

            # Only run SAM on boxes that are NOT already refined
            to_refine_mask = ~already_refined
            if to_refine_mask.any():
                boxes_to_refine = boxes[to_refine_mask]
                boxes_device = boxes_to_refine.to(sam.device)
                sam_box_prompts = resize_transform.apply_boxes_torch(boxes_device, img_shape)

                sam_batch_size = 50
                sam_scores_list = []
                sam_refined_list = []

                for k in range(0, len(sam_box_prompts), sam_batch_size):
                    batch = sam_box_prompts[k:k+sam_batch_size]
                    if len(batch) == 0:
                        continue

                    batched_input = [{
                        "image": curr_image,
                        "boxes": batch,
                        "original_size": img_shape
                    }]

                    if tracker:
                        t_start_sam = tracker.log_module_start("SAM")

                    out = sam(batched_input, multimask_output=False)

                    masks = out[0]['masks'].clone().detach()
                    refined_boxes = batched_mask_to_box(masks).squeeze(1)
                    iou_preds = out[0]['iou_predictions']
                    sam_refined_list.append(refined_boxes)
                    sam_scores_list.append(iou_preds)

                    if tracker:
                        tracker.log_module_end("SAM", t_start_sam, metadata={
                            "num_masks": len(masks),
                            "coverage": masks.float().mean().item() * 100,
                            "_device_probe": refined_boxes  # tensor for device detection
                        })

                    del out, masks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                if len(sam_scores_list) > 0:
                    sam_refined[to_refine_mask] = torch.cat(sam_refined_list, dim=0)
                    sam_scores[to_refine_mask] = torch.cat(sam_scores_list, dim=0).squeeze(1)

            # Do NOT overwrite boxes yet, we will select better box later.

            if cache_available and cache_path and len(sam_scores) > 0:
                temp_path = cache_path + ".tmp"
                try:
                    os.makedirs(sam_cache_dir, exist_ok=True)
                    torch.save({
                        'sam_scores': sam_scores,
                        'sam_refined': sam_refined,
                        'boxes_count': len(boxes),
                        'boxes_hash': boxes_hash,
                        'image_id': image_id
                    }, temp_path)
                    os.replace(temp_path, cache_path)
                    print(f"[SAM Cache] Saved: {os.path.basename(cache_path)}")
                except Exception as e:
                    print(f"[SAM Cache] Save failed: {e}")
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

        except Exception as e:
            if verbose:
                print(f"[SAM] Warning: {e}")
            sam_scores = None

    # Select better box and score from SAM
    if sam is not None and sam_refined is not None and sam_scores is not None and len(sam_refined) == len(boxes):
        # Always use the SAM-refined boxes and their predicted IoU scores
        # for SRM scaling, exactly like the COCO pipeline.
        if already_refined is not None:
            to_refine_mask = ~already_refined
        else:
            to_refine_mask = torch.ones(len(boxes), dtype=torch.bool, device=device)
            
        final_boxes = boxes.clone()
        final_boxes[to_refine_mask] = sam_refined[to_refine_mask]
        
        boxes = final_boxes
        # Keep sam_scores as predicted by SAM
    elif sam is not None and sam_refined is not None:
        boxes = sam_refined

    if srm is not None and sam_scores is not None and len(sam_scores) == len(scores):
        try:
            if tracker:
                t_start_srm = tracker.log_module_start("SRM")

            orig_mean = scores.mean().item()
            
            # Apply SRM ONLY to boxes that are NOT already refined
            to_refine_mask = ~already_refined
            if to_refine_mask.any():
                refined_scores_subset = srm.refine_scores(scores[to_refine_mask], sam_scores[to_refine_mask])
                scores[to_refine_mask] = refined_scores_subset

            new_mean = scores.mean().item()

            if tracker:
                tracker.log_module_end("SRM", t_start_srm, metadata={
                    "score_delta": {
                        "before_mean": orig_mean,
                        "after_mean": new_mean,
                        "change_percent": (new_mean - orig_mean) / (orig_mean + 1e-6) * 100
                    }
                })
        except Exception as e:
            if verbose:
                print(f"[SRM] Warning: {e}")

    # Step 5: Frequency-Aware Calibration + Logit Adjustment
    # Apply calibration ONLY to novel proposals (not already_refined known boxes).
    # Known boxes come from Mask-RCNN with calibrated classification scores —
    # applying LVIS rare/common/frequent boosts to them distorts their ranking.
    if calibrator is not None and len(scores) > 0:
        if already_refined is not None and already_refined.any() and (~already_refined).any():
            novel_calib_mask = ~already_refined
            novel_scores_calib = calibrator.calibrate(
                scores[novel_calib_mask], classes[novel_calib_mask],
                apply_logit_adjustment=True, tau=tau
            )
            scores = scores.clone()
            scores[novel_calib_mask] = novel_scores_calib
        elif already_refined is None or not already_refined.any():
            # No known boxes — calibrate everything
            scores = calibrator.calibrate(scores, classes, apply_logit_adjustment=True, tau=tau)
        # else: all known boxes — skip calibration entirely

    # Ensure all tensors are on the same device
    boxes = boxes.float().to(device)
    scores = scores.float().to(device)
    classes = classes.long().to(device)

    if is_voc and len(classes) > 0:
        # Keep only VOC classes [0, 19]
        voc_mask = (classes >= 0) & (classes < 20)
        boxes = boxes[voc_mask]
        scores = scores[voc_mask]
        classes = classes[voc_mask]

    # Post-SAM deduplication (APF WBF or standard NMS)
    if apf is not None:
        dedup = apf.deduplicate(boxes, scores, classes)
        boxes = dedup["boxes"]
        scores = dedup["scores"]
        classes = dedup["classes"]
    else:
        keep = batched_nms(boxes, scores, classes, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]

    # Top-K matches max_dets_per_image in evaluator
    if len(scores) > max_dets_per_image:
        topk_scores, topk_idxs = torch.topk(scores, max_dets_per_image)
        boxes = boxes[topk_idxs]
        scores = topk_scores
        classes = classes[topk_idxs]

    return boxes.cpu(), scores.cpu(), classes.cpu()