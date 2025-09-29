#!/bin/bash
set -e

# Optional: update core + key custom nodes
git -C /root/ComfyUI pull || true
git -C /root/ComfyUI/custom_nodes/ComfyUI-KJNodes pull || true
git -C /root/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper pull || true

# Start JupyterLab (no token)
jupyter lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --LabApp.token='' --LabApp.password='' &

# Start ComfyUI
python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --use-sage-attention \
    --fast \
    --preview-method latent2rgb
