#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    print(f"Starting Local Server from project root: {project_root}")
    
    # Set environment context
    os.environ["NOD_ENV"] = "local"
    
    # Insert project root to python path
    sys.path.insert(0, str(project_root))
    
    try:
        import uvicorn
    except ImportError:
        print("Uvicorn not found in current environment. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn"])
        
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
        print("\nStopping local server.")

if __name__ == "__main__":
    main()
