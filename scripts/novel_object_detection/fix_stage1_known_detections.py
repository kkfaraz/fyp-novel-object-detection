"""
Fix Stage 1 cache: Replace empty known_boxes with correct RCNN detections.

The original pipeline loaded LVIS-pretrained Mask-RCNN weights that had major
architecture mismatches (FPN, RPN, ROI heads), producing 0 known detections
on every image.

This script re-runs inference using correct COCO-pretrained Mask-RCNN (80 classes)
and updates each cached Stage 1 file with non-empty known_boxes/known_scores/known_labels.
"""

import os, sys, pickle, json, gc, time, argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from gpu_utils import clear_gpu_memory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str, default='cache/stage1_lvis')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=None, help='Limit for testing')
    parser.add_argument('--image-list', type=str, default=None, help='File with image filenames to process')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(__file__))
    from load_models import load_fully_supervised_trained_model
    with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
        params = json.load(f)

    proj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    cfg_file = os.path.join(proj_path, params['cfg_file'])
    rcnn_weight_path = os.path.join(proj_path, params['rcnn_weight_dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"[Fix] Loading Mask-RCNN with correct COCO weights...")
    rcnn_model, cfg = load_fully_supervised_trained_model(cfg_file, rcnn_weight_path, device=device)
    rcnn_model.eval()
    print(f"[Fix] Model loaded with {cfg.model.roi_heads.box_predictor.num_classes} classes")

    from detectron2.data.transforms import ResizeShortestEdge
    from detectron2.data import MetadataCatalog
    from utils import get_coco_to_lvis_mapping
    lvis_data_split = params.get("lvis_data_split", params.get("data_split", "lvis_v1_val"))
    coco_to_lvis = get_coco_to_lvis_mapping(cfg, lvis_data_split)

    cache_dir = os.path.join(proj_path, args.cache_dir)
    if not os.path.isdir(cache_dir):
        print(f"[Fix] ERROR: Cache directory not found: {cache_dir}")
        return

    cache_files = sorted([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])

    if args.image_list:
        with open(args.image_list) as f:
            allowed = set(line.strip() for line in f)
        cache_files = [f for f in cache_files if f in allowed]

    if args.max_images:
        cache_files = cache_files[:args.max_images]

    print(f"[Fix] Processing {len(cache_files)} cache files...")

    total_updated = 0
    total_skipped = 0
    total_errors = 0

    torch.backends.cudnn.benchmark = True

    transform = ResizeShortestEdge(800, 1333)
    
    for idx, fname in enumerate(tqdm(cache_files, desc="Updating cache")):
        fpath = os.path.join(cache_dir, fname)
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)

            meta = data.get('meta', {})
            img_path = meta.get('filename', '')
            if not img_path or not os.path.exists(img_path):
                total_skipped += 1
                continue

            existing_known = len(data.get('known_boxes', []))
            if existing_known > 0:
                total_skipped += 1
                continue

            img = np.array(Image.open(img_path).convert('RGB'))
            h, w = img.shape[:2]
            resized = transform.get_transform(img).apply_image(img)
            resized_t = torch.as_tensor(resized.astype('float32').transpose(2, 0, 1))

            inputs = [{'image': resized_t.to(device), 'height': h, 'width': w}]

            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = rcnn_model(inputs)

            instances = outputs[0]['instances']
            known_mask = instances.pred_classes < 80
            known_boxes = instances.pred_boxes.tensor[known_mask].cpu().numpy()
            known_scores = instances.scores[known_mask].cpu().numpy()
            known_classes = instances.pred_classes[known_mask].cpu().numpy()

            known_classes_lvis = np.array([coco_to_lvis[c] for c in known_classes], dtype=np.int32)

            data['known_boxes'] = known_boxes.astype(np.float32)
            data['known_scores'] = known_scores.astype(np.float32)
            data['known_labels'] = known_classes_lvis.astype(np.int32)

            temp_path = fpath + '.tmp'
            with open(temp_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_path, fpath)

            total_updated += 1

        except Exception as e:
            total_errors += 1
            if total_errors <= 3:
                print(f"[Fix] ERROR on {fname}: {e}")
            continue

        # Periodic cleanup: every 500 images
        if (idx + 1) % 500 == 0:
            clear_gpu_memory()

    # Final cleanup
    clear_gpu_memory()

    print(f"\n[Fix] Done! Updated: {total_updated}, Skipped: {total_skipped}, Errors: {total_errors}")

if __name__ == '__main__':
    main()
