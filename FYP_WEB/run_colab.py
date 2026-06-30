#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def setup_tunnel(port: int):
    authtoken = os.environ.get("NGROK_AUTHTOKEN")
    if not authtoken:
        print("[Colab Runner] No NGROK_AUTHTOKEN found in environment variables.")
        print("[Colab Runner] Running server on port 8000 without tunnel.")
        return None

    try:
        from pyngrok import ngrok
    except ImportError:
        print("[Colab Runner] pyngrok not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyngrok"])
        from pyngrok import ngrok

    print(f"[Colab Runner] Setting ngrok authtoken...")
    ngrok.set_auth_token(authtoken)
    
    tunnel = ngrok.connect(port)
    print(f"\n==================================================")
    print(f"[Colab Tunnel] Public URL: {tunnel.public_url}")
    print(f"==================================================\n")
    return tunnel

def main():
    project_root = Path(__file__).resolve().parent
    print(f"Starting Colab Server from project root: {project_root}")
    
    # Force context
    os.environ["NOD_ENV"] = "colab"
    os.environ["PYTHONPATH"] = str(project_root)
    sys.path.insert(0, str(project_root))
    
    port = 8000
    tunnel = setup_tunnel(port)
    
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--log-level",
        "info"
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
    except KeyboardInterrupt:
        print("\nStopping Colab server.")
    finally:
        if tunnel:
            print("[Colab Tunnel] Closing ngrok tunnel...")
            from pyngrok import ngrok
            ngrok.disconnect(tunnel.public_url)

if __name__ == "__main__":
    main()
