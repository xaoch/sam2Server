# FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
FROM registry.cloud.rt.nyu.edu/sw77/cuda:12.6.3-ubuntu-22.04-20250103

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX"

WORKDIR /app

# Install system dependencies, including libglib2.0-0 needed by cv2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python3-venv python3-distutils \
        build-essential cmake ninja-build git curl \
        libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# Copy code

# Install Python dependencies
RUN python3 -m pip install numpy
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# If you installed opencv-python, ensure system libs are present; or consider headless:
RUN python3 -m pip install opencv-python-headless

COPY . . 

# Editable install
RUN python3 -m pip install -e .

# Install FastAPI & Uvicorn
RUN python3 -m pip install fastapi uvicorn python-multipart


WORKDIR /app/server
EXPOSE 4040
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "4040"]