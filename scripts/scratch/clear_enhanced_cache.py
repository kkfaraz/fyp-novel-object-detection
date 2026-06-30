"""
Clear stale stage2_boxes/scores/classes from VLRM cache files.
Preserves the VLRM captions (expensive to regenerate) but removes
the pre-computed stage2 outputs that were generated with old parameters
(threshold=0.20, fusion_alpha=0.4).

This forces the pipeline to re-run Mahalanobis scoring with the new params
(threshold=0.10, fusion_alpha=0.55) while reusing cached captions.
"""
import os
import pickle
import sys
from tqdm import tqdm

cache_dir = "cache/vlrm_outputs_lvis"
if not os.path.isdir(cache_dir):
    print(f"Cache dir not found: {cache_dir}")
    sys.exit(1)

files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
print(f"Scanning {len(files)} cache files in {cache_dir}...")

cleaned = 0
errors = 0

for fname in tqdm(files, desc="Cleaning enhanced cache"):
    fpath = os.path.join(cache_dir, fname)
    try:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        
        # Only modify files that have stale stage2 outputs
        if "stage2_boxes" in data:
            del data["stage2_boxes"]
            del data["stage2_scores"]
            del data["stage2_classes"]
            
            temp = fpath + ".tmp"
            with open(temp, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp, fpath)
            cleaned += 1
    except Exception as e:
        errors += 1
        if errors < 5:
            print(f"  Error on {fname}: {e}")

print(f"\nDone: cleaned {cleaned} files, {errors} errors, {len(files) - cleaned - errors} already clean")
