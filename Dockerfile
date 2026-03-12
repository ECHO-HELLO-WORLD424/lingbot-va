# LingBot-VA Docker Image
#
# NOTE: cuRobo is NOT built in this image — install it manually after mounting the project:
#   cd robotwin/envs/curobo && pip install -e . --no-build-isolation


FROM harbor.local.clusters/infrawaves/pytorch:24.10-py3

# FROM nvidia/cuda:12.6.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    libncurses5-dev \
    tk-dev \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libgles2-mesa \
    libglvnd0 \
    libglvnd-dev \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    unzip \
    tmux \
    nvtop \
    btop \
    python3.10-venv \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && mkdir -p /usr/share/vulkan/icd.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json

RUN pip install --upgrade pip setuptools wheel

# PyTorch 2.9.0 (CUDA 12.6)
RUN pip install \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu126

# LingBot-VA core dependencies
RUN pip install \
    numpy==1.26.4 \
    einops \
    diffusers==0.36.0 \
    transformers==4.55.2 \
    "tokenizers>=0.21.4" \
    accelerate \
    safetensors \
    Pillow \
    "imageio[ffmpeg]==2.34.2" \
    tqdm \
    scipy \
    websockets \
    msgpack \
    opencv-python \
    matplotlib \
    ftfy \
    easydict \
    lerobot==0.3.3 \
    wandb

# flash-attn (long compile ~60-90 min)
RUN pip install flash-attn --no-build-isolation

# RoboTwin dependencies
RUN pip install \
    "transforms3d==0.4.2" \
    "sapien==3.0.0b1" \
    "mplib==0.2.1" \
    "gymnasium==0.29.1" \
    "trimesh==4.4.3" \
    "open3d==0.18.0" \
    pydantic \
    zarr \
    openai \
    "huggingface_hub==0.36.2" \
    h5py \
    "azure==4.0.0" \
    azure-ai-inference \
    "pyglet<2" \
    moviepy \
    termcolor \
    av

# Patch sapien: add encoding="utf-8" to urdf_loader.py
RUN sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
    /usr/local/lib/python3.10/dist-packages/sapien/wrapper/urdf_loader.py

# Patch mplib: remove spurious collision check in screw planner
RUN sed -i -E \
    's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
    /usr/local/lib/python3.10/dist-packages/mplib/planner.py

# Patch cv2: comment out broken LayerId line in typing/__init__.py (line 162)
RUN sed -i '162s/^/# /' \
    /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py

# pytorch3d from source (must match exact torch ABI, long compile ~30-60 min)
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation

# Workspace
WORKDIR /workspace

CMD ["/bin/bash"]
