#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    print(f"Starting Kaggle Server from project root: {project_root}")
    
    # Force context
    os.environ["NOD_ENV"] = "kaggle"
    sys.path.insert(0, str(project_root))
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        "info"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping Kaggle server.")

if __name__ == "__main__":
    main()
