import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import numpy as np
    print(f"numpy: SUCCESS (version {np.__version__})")
except ImportError:
    print("numpy: FAILED")

try:
    import torch
    print(f"torch: SUCCESS (version {torch.__version__})")
    print(f"CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("torch: FAILED")

try:
    import fastapi
    print(f"fastapi: SUCCESS (version {fastapi.__version__})")
except ImportError:
    print("fastapi: FAILED")

try:
    import detectron2
    print("detectron2: SUCCESS")
except ImportError:
    print("detectron2: FAILED")
