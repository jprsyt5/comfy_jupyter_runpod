from huggingface_hub import hf_hub_download, snapshot_download
import os

def download_models():
    cache_dir = os.environ.get("HF_HOME", "/root/.cache/huggingface")
    base_dir = "/root/ComfyUI/models"
    HF_TOKEN = os.environ.get("HF_TOKEN")

    files_to_download = [
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/text_encoders/umt5_xxl_fp16.safetensors", "text_encoders"),
        ("Kijai/WanVideo_comfy", "umt5-xxl-enc-bf16.safetensors", "text_encoders"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/clip_vision/clip_vision_h.safetensors", "clip_vision"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/diffusion_models/wan2.2_animate_14B_bf16.safetensors", "diffusion_models"),
        ("Comfy-Org/Wan_2.2_ComfyUI_Repackaged", "split_files/loras/wan2.2_animate_14B_relight_lora_bf16.safetensors", "loras"),
        ("Kijai/sam2-safetensors", "sam2.1_hiera_base_plus.safetensors", "sam2"),
        ("Kijai/sam2-safetensors", "sam2.1_hiera_large.safetensors", "sam2"),
        ("Kijai/WanVideo_comfy", "Fun/VACE/Wan2_2_Fun_VACE_module_A14B_HIGH_bf16.safetensors", "diffusion_models"),
        ("Kijai/WanVideo_comfy", "Fun/VACE/Wan2_2_Fun_VACE_module_A14B_LOW_bf16.safetensors", "diffusion_models"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/old/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_HIGH_Lightning_4steps_lora_250928_rank128_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "LoRAs/Wan22-Lightning/Wan22_A14B_T2V_LOW_Lightning_4steps_lora_250928_rank64_fp16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank256_bf16.safetensors", "loras"),
        ("Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors", "loras"),
        ("yuvraj108c/ComfyUI-Upscaler-Onnx", "4x-ClearRealityV1.onnx", "onnx"),
        ("yuvraj108c/ComfyUI-Upscaler-Onnx", "4x_foolhardy_Remacri.onnx", "onnx"),
        ("yuvraj108c/ComfyUI-Upscaler-Onnx", "4x-UltraSharpV2_Lite.onnx", "onnx"),
        ("jayn7/upscaler", "1x-ITF-SkinDiffDetail-Lite-v1.onnx", "onnx"),
        ("gemasai/x1_ITF_SkinDiffDetail_Lite_v1", "x1_ITF_SkinDiffDetail_Lite_v1.pth", "upscale_models"),
        ("jayn7/rife-tensorrt", "rife47_ensemble_True_scale_1_sim.engine", "tensorrt/rife"),
        ("jayn7/rife-tensorrt", "rife48_ensemble_True_scale_1_sim.engine", "tensorrt/rife"),
        ("jayn7/rife-tensorrt", "rife49_ensemble_True_scale_1_sim.engine", "tensorrt/rife"),
    ]

    repos_to_download = [
        ("jayn7/wan-lora-22", "loras")
    ]

    for repo, filename, subfolder in files_to_download:
        print(f"⬇️ {repo} :: {filename}")
        real_path = hf_hub_download(repo_id=repo, filename=filename, cache_dir=cache_dir, token=HF_TOKEN)
        target_dir = os.path.join(base_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        link_path = os.path.join(target_dir, os.path.basename(filename))
        if os.path.islink(link_path) or os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(real_path, link_path)
        print(f"✅ {link_path} -> {real_path}")

    for repo_id, subfolder in repos_to_download:
        print(f"⬇️ snapshot: {repo_id}")
        repo_path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, token=HF_TOKEN)
        target_dir = os.path.join(base_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)
        for root, _, files in os.walk(repo_path):
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(target_dir, f)
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
                print(f"✅ {dst} -> {src}")

    print(f"\n🎉 All models symlinked under {base_dir}")

if __name__ == "__main__":
    download_models()
