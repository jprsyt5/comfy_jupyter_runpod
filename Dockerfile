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

# Custom nodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone https://github.com/rgthree/rgthree-comfy.git /root/ComfyUI/custom_nodes/rgthree-comfy && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git /root/ComfyUI/custom_nodes/ComfyUI_essentials && \
    git clone https://github.com/crystian/ComfyUI-Crystools.git /root/ComfyUI/custom_nodes/ComfyUI-Crystools && \
    cd /root/ComfyUI/custom_nodes/ComfyUI-Crystools && pip install -r requirements.txt && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git /root/ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    cd /root/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git /root/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper && \
    git clone https://github.com/Extraltodeus/Skimmed_CFG.git /root/ComfyUI/custom_nodes/Skimmed_CFG && \
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /root/ComfyUI/custom_nodes/comfyui_controlnet_aux && \
    git clone https://github.com/o-l-l-i/ComfyUI-Olm-DragCrop.git /root/ComfyUI/custom_nodes/ComfyUI-Olm-DragCrop && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2.git /root/ComfyUI/custom_nodes/ComfyUI-segment-anything-2 && \
    git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git /root/ComfyUI/custom_nodes/ComfyUI-WanAnimatePreprocess && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use.git /root/ComfyUI/custom_nodes/ComfyUI-Easy-Use && \
    cd /root/ComfyUI/custom_nodes/ComfyUI-Easy-Use && pip install -r requirements.txt && \
    git clone -b 1x_upscaler_support https://github.com/jprsyt5/ComfyUI-Upscaler-Tensorrt.git /root/ComfyUI/custom_nodes/ComfyUI-Upscaler-Tensorrt && \
    cd /root/ComfyUI/custom_nodes/ComfyUI-Upscaler-Tensorrt && pip install polygraphy requests && \
    git clone https://github.com/yuvraj108c/ComfyUI-Rife-Tensorrt.git /root/ComfyUI/custom_nodes/ComfyUI-Rife-Tensorrt && \
    git clone https://github.com/M1kep/Comfy_KepListStuff.git /root/ComfyUI/custom_nodes/Comfy_KepListStuff

# Create dirs
RUN mkdir -p /root/ComfyUI/models /root/ComfyUI/output /root/.cache/huggingface

# Copy downloader + entrypoint
COPY comfyjupiter-download.py /root/comfyjupiter-download.py
COPY entrypoint.sh /root/ComfyUI/entrypoint.sh
RUN chmod +x /root/ComfyUI/entrypoint.sh

WORKDIR /root/ComfyUI
ENV HF_HUB_ENABLE_HF_TRANSFER=1

EXPOSE 8888 8188

CMD ["/root/ComfyUI/entrypoint.sh"]
