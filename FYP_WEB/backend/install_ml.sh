#!/bin/bash
set -e

echo "============================================="
echo " Cooperative NOD Backend Automated Setup"
echo "============================================="

VENV_DIR=~/nod_backend_env
PROJECT_DIR="/media/farazkk/Haseeb Butt/Fyp/cooperative-foundational-models"

# 1. Ensure venv exists
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
fi

# 2. Activate venv
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# 3. Install PyTorch + Torchaudio + Torchvision (CUDA 11.8)
echo "Installing PyTorch & CUDA... (this takes ~10-15 mins depending on internet speed)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --timeout=3000

# 4. Install FastAPI backend requirements
echo "Installing Backend Requirements..."
cd "$PROJECT_DIR/frontend/backend"
pip install -r requirements.txt

# 5. Install other ML requirements (huggingface, opencv, groundingdino etc)
echo "Installing Additional Dependencies..."
cd "$PROJECT_DIR"
pip install -r captioner/requirements.txt || echo "Optional captioner reqs failed, continuing"
pip install opencv-python transformers supervision pycocotools scikit-learn

echo "Checking for Detectron2..."
python -c "import detectron2" 2>/dev/null || python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

echo "============================================="
echo " Setup Complete. Starting API Server..."
echo "============================================="
cd "$PROJECT_DIR/frontend/backend"
python server.py
