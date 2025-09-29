ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

FROM python:3.12-slim

# System deps
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
    build-essential wget gnupg ca-certificates lsb-release \
    pkg-config cmake libcairo2-dev libpango1.0-dev && \
    rm -rf /var/lib/apt/lists/*

# CUDA 12.4 (same as Modal)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-compiler-12-4 cuda-libraries-dev-12-4 && \
    echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> /etc/profile && \
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /etc/profile && \
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /etc/profile

# Upgrade pip + install essentials
RUN python -m pip install --upgrade pip setuptools wheel
# Torch (CUDA 12.9 wheel, works with H100)
RUN pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129

# Core deps
RUN pip install \
    huggingface_hub opencv-python imageio-ffmpeg ftfy \
    "accelerate>=1.2.1" einops "diffusers>=0.33.0" "peft>=0.15.0" \
    "sentencepiece>=0.2.0" protobuf pyloudnorm triton==3.4.0 \
    jupyter gguf hf-transfer

# Extra deps
RUN pip install \
    importlib_metadata scipy filelock Pillow pyyaml scikit-image \
    python-dateutil mediapipe svglib fvcore yapf omegaconf \
    addict yacs "trimesh[easy]" albumentations scikit-learn matplotlib \
    colored cuda-python "cupy-cuda12x>=13.3.0" easydict

# SageAttention wheel
RUN pip install https://huggingface.co/jayn7/SageAttention_wheel/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl?download=true

# Clone ComfyUI core
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /root/ComfyUI && \
    cd /root/ComfyUI && pip install -r requirements.txt

# Custom nodes (shallow clone + install requirements)
RUN git clone --depth=1 https://github.com/kijai/ComfyUI-KJNodes.git /root/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone --depth=1 https://github.com/rgthree/rgthree-comfy.git /root/ComfyUI/custom_nodes/rgthree-comfy && \
    git clone --depth=1 https://github.com/cubiq/ComfyUI_essentials.git /root/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone --depth=1 https://github.com/crystian/ComfyUI-Crystools.git /root/ComfyUI/custom_nodes/ComfyUI-Crystools && \
    pip install -r /root/ComfyUI/custom_nodes/ComfyUI-Crystools/requirements.txt && \
    git clone --depth=1 https://github.com/ltdrdata/ComfyUI-Manager.git /root/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone --depth=1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    pip install -r /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt && \
    git clone --depth=1 https://github.com/Extraltodeus/Skimmed_CFG.git /root/ComfyUI/custom_nodes/Skimmed_CFG && \
    git clone --depth=1 https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop.git /root/ComfyUI/custom_nodes/ComfyUI-Olm-DragCrop && \
    git clone --depth=1 https://github.com/yolain/ComfyUI-Easy-Use.git /root/ComfyUI/custom_nodes/ComfyUI-Easy-Use && \
    pip install -r /root/ComfyUI/custom_nodes/ComfyUI-Easy-Use/requirements.txt && \
    git clone --depth=1 -b 1x_upscaler_support https://github.com/jprsyt5/ComfyUI-Upscaler-Tensorrt.git /root/ComfyUI/custom_nodes/ComfyUI-Upscaler-Tensorrt && \
    pip install polygraphy requests || true && \
    git clone --depth=1 https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt.git /root/ComfyUI/custom_nodes/ComfyUI-Rife-Tensorrt && \
    git clone --depth=1 https://github.com/M1kep/Comfy_KepListStuff.git /root/ComfyUI/custom_nodes/Comfy_KepListStuff && \
    # cleanup .git folders to save space
    find /root/ComfyUI/custom_nodes -name ".git" -type d -exec rm -rf {} +

# Create dirs
RUN mkdir -p /root/ComfyUI/models /root/ComfyUI/output /root/.cache/huggingface
RUN rm -rf /root/.cache/pip

# Copy downloader + entrypoint
COPY install_tensorrt.sh /root/ComfyUI/install_tensorrt.sh
RUN chmod +x /root/ComfyUI/install_tensorrt.sh
COPY comfyjupiter-download.py /root/comfyjupiter-download.py
COPY entrypoint.sh /root/ComfyUI/entrypoint.sh
RUN chmod +x /root/ComfyUI/entrypoint.sh

WORKDIR /root/ComfyUI
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8888 8188

CMD ["/root/ComfyUI/entrypoint.sh"]
