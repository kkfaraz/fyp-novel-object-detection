FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    opencv-python \
    pillow \
    pyngrok \
    open_clip_torch \
    pycocotools

# Install Segment Anything (SAM)
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Clone and install Grounding DINO if needed, or rely on local builds.
# Assuming standard groundingdino package is installed or package is provided.
RUN pip install --no-cache-dir git+https://github.com/IDEA-Research/GroundingDINO.git

# Copy codebase
COPY . /workspace

EXPOSE 8000

CMD ["python3", "run_local.py"]
