"""
CLI Entry point for single-image or custom inference using the CoNOD pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root and scripts directory to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJ_ROOT / "scripts"
NOD_DIR = SCRIPTS_DIR / "novel_object_detection"

for p in [str(PROJ_ROOT), str(SCRIPTS_DIR), str(NOD_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config_utils import load_params, resolve_path
from gpu_utils import get_device


def main():
    parser = argparse.ArgumentParser(description="CoNOD Novel Object Detection Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", help="Directory to save output visualizations/predictions")
    parser.add_argument("--config", type=str, default="scripts/novel_object_detection/params.json", help="Path to params.json configuration file")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on ('cuda' or 'cpu')")
    parser.add_argument("--no-visualize", action="store_true", help="Disable saving visualization images")

    args = parser.parse_args()

    # Load configuration
    params = load_params(args.config)
    
    # Resolve paths
    image_path = resolve_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image file not found: {image_path}")

    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = args.device if args.device else str(get_device())
    print(f"[Inference] Running on image: {image_path}")
    print(f"[Inference] Output directory: {out_dir}")
    print(f"[Inference] Device: {device}")

    # Import pipeline setup and single image inference
    from inference_single_image import setup, inference_single_image

    gdino_checkpoint = params.get("gdino_checkpoint_resolved", params.get("gdino_checkpoint", "GDINO_weights.pth"))
    cfg_file = params.get("cfg_file_resolved", params.get("cfg_file"))
    rcnn_weight_dir = params.get("rcnn_weight_dir_resolved", params.get("rcnn_weight_dir"))
    sam_checkpoint = params.get("sam_checkpoint_resolved", params.get("sam_checkpoint", "SAM_weights.pth"))
    class_len_per_prompt = params.get("class_len_per_prompt", 81)

    model, text_prompt_list, param_dict = setup(
        outputs_dir=str(out_dir),
        gdino_checkpoint=gdino_checkpoint,
        cfg_file=cfg_file,
        rcnn_weight_dir=rcnn_weight_dir,
        sam_checkpoint=sam_checkpoint,
        class_len_per_prompt=class_len_per_prompt
    )

    param_dict["visualize"] = not args.no_visualize
    param_dict["device"] = device

    print("[Inference] Executing pipeline...")
    inference_single_image(model, str(image_path), text_prompt_list, param_dict)
    print(f"[Inference] ✓ Results successfully saved to {out_dir}")


if __name__ == "__main__":
    main()
