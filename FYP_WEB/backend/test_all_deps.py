import sys
import os

# Create an exhaustive import test script
try:
    import open_clip
    import torch
    import cv2
    import transformers
    import detectron2
    import supervision
    import groundingdino
    import segment_anything
    import lvis
    print("ALL DEPENDENCIES PRESENT.")
except Exception as e:
    print(f"MISSING DEPENDENCY: {e}")
