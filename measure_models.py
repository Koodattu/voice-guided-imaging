import os
import time
import psutil
import pynvml
import torch
from PIL import Image

from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableVideoDiffusionPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler
)

# ------------------------------------------------------------------
# 1) OPTIONAL: If needed for private models, authenticate here
# ------------------------------------------------------------------
# Uncomment and set your Hugging Face token if accessing private models
# login("YOUR_HUGGINGFACE_TOKEN")

# ------------------------------------------------------------------
# 2) GPU and Cache Setup
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "./model_cache"
os.makedirs(cache_dir, exist_ok=True)
print(f"Using device: {device}")

# ------------------------------------------------------------------
# 3) Initialize NVML (for GPU VRAM measurement)
# ------------------------------------------------------------------
pynvml.nvmlInit()
gpu_handle = None
try:
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0
except pynvml.NVMLError as e:
    print(f"Error accessing GPU 0: {e}")

# ------------------------------------------------------------------
# 4) Memory / Time Measurement Utilities
# ------------------------------------------------------------------
def get_ram_usage_gb():
    """Returns used RAM and total RAM in GB."""
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024**3)
    total_gb = vm.total / (1024**3)
    return used_gb, total_gb

def get_vram_usage_gb(handle):
    """Returns used VRAM and total VRAM in GB for given GPU handle."""
    if not handle:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / (1024**3)
        total_gb = mem_info.total / (1024**3)
        return used_gb, total_gb
    except pynvml.NVMLError as error:
        print(f"Error retrieving VRAM usage: {error}")
        return None, None

def print_usage(stage):
    """Prints RAM and VRAM usage at a given stage."""
    used_ram, total_ram = get_ram_usage_gb()
    used_vram, total_vram = get_vram_usage_gb(gpu_handle)

    if used_vram is not None:
        print(
            f"{stage} - "
            f"RAM: {used_ram:.2f}/{total_ram:.2f} GB | "
            f"VRAM: {used_vram:.2f}/{total_vram:.2f} GB"
        )
    else:
        print(
            f"{stage} - "
            f"RAM: {used_ram:.2f}/{total_ram:.2f} GB | "
            f"VRAM: N/A"
        )

def measure_time_and_usage(action_name, func, *args, **kwargs):
    """
    Helper function to:
      - Print usage before action
      - Time the function call
      - Print usage after action
      - Print elapsed time
    """
    print_usage(f"Before {action_name}")
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    print_usage(f"After {action_name}")
    print(f"Time for {action_name}: {elapsed:.2f} seconds\n")
    return result

# ------------------------------------------------------------------
# 5) Model Load / Unload Functions
#    (Adapted from your reference code)
# ------------------------------------------------------------------
def load_sdxl_lightning():
    """
    Loads SDXL-Lightning using the same approach as in your reference.
    Returns the pipeline.
    """
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"
    
    print("Loading SDXL-Lightning model...")
    
    # Load UNet
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
    
    # Create pipeline
    txt2img = StableDiffusionXLPipeline.from_pretrained(
        base, 
        unet=unet, 
        torch_dtype=torch.float16, 
        variant="fp16", 
        cache_dir=cache_dir
    )
    txt2img.to(device)
    txt2img.scheduler = EulerDiscreteScheduler.from_config(
        txt2img.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    # Optional memory optimizations
    txt2img.enable_model_cpu_offload()
    txt2img.enable_vae_slicing()
    
    print("SDXL-Lightning model loaded successfully!")
    return txt2img

def load_instruct_pix2pix():
    """
    Loads timbrooks/instruct-pix2pix using the same approach as in your reference.
    Returns the pipeline.
    """
    print("Loading Instruct-Pix2Pix model...")
    pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        safety_checker=None
    )
    pix2pix.to(device)
    pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(pix2pix.scheduler.config)
    pix2pix.enable_model_cpu_offload()
    pix2pix.enable_vae_slicing()
    print("Instruct-Pix2Pix model loaded successfully!")
    return pix2pix

def load_video_diffusion():
    """
    Loads stabilityai/stable-video-diffusion-img2vid-xt-1-1 using the same approach as in your reference.
    Returns the pipeline.
    """
    print("Loading Stable Video Diffusion model...")
    img2vid = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_dir
    )
    img2vid.to(device)
    img2vid.enable_model_cpu_offload()
    
    # Some additional config for chunking:
    if img2vid.unet:
        img2vid.unet.enable_forward_chunking()
    
    print("Stable Video Diffusion model loaded successfully!")
    return img2vid

def unload_model(model):
    """
    Unloads the model from VRAM and clears CUDA cache, same as your reference approach.
    """
    print(f"Unloading model: {type(model).__name__}")
    del model
    torch.cuda.empty_cache()
    print("Model unloaded successfully!")

# ------------------------------------------------------------------
# 6) Simple Generation Functions
# ------------------------------------------------------------------
def generate_sdxl_lightning(pipe, output_path="throwaway-image.png"):
    """
    Generates a single image using the loaded SDXL-Lightning pipeline.
    Saves the image to the specified path.
    """
    prompt = "A beautiful landscape painting of mountains, trending on artstation"
    print(f"Generating image with SDXL-Lightning for prompt: '{prompt}'")
    image = pipe(prompt, num_inference_steps=4, guidance_scale=7.5).images[0]
    image.save(output_path)
    print(f"Image saved as '{output_path}'")
    return output_path

def generate_instruct_pix2pix(pipe, input_image_path="throwaway-image.png", output_image_path="edited-image.png"):
    """
    Generates an edited image using the loaded Instruct-Pix2Pix pipeline.
    Saves the edited image to the specified path.
    """
    prompt = "Make the cat wear sunglasses"
    print(f"Generating edited image with Instruct-Pix2Pix for prompt: '{prompt}'")
    
    # Load the input image
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    # Perform the edit
    edited_image = pipe(prompt=prompt, image=init_image, num_inference_steps=4, guidance_scale=7.5).images[0]
    edited_image.save(output_image_path)
    print(f"Edited image saved as '{output_image_path}'")
    return output_image_path

def generate_video_diffusion(pipe, input_image_path="edited-image.png", output_video_path="generated_video.mp4"):
    """
    Generates a short video using the loaded Stable Video Diffusion pipeline.
    Saves the video to the specified path.
    """
    prompt = "Animate the landscape with moving clouds and flowing rivers"
    print(f"Generating video with Stable Video Diffusion for prompt: '{prompt}'")
    
    # Load the input image
    init_image = Image.open(input_image_path).convert("RGB").resize((512, 512))
    
    # Perform the video generation
    video_frames = pipe(init_image, decode_chunk_size=2, num_inference_steps=10,).frames
    
    # Save the video using moviepy
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip([frame for frame in video_frames], fps=7)
    clip.write_videofile(output_video_path, codec="libx264")
    print(f"Video saved as '{output_video_path}'")
    return output_video_path

# ------------------------------------------------------------------
# 7) Main Script Flow
# ------------------------------------------------------------------
def main():
    print("\n=== Measuring resources and timings for SDXL-Lightning, Instruct-Pix2Pix, and Stable Video Diffusion ===\n")

    # ================ 1) SDXL-Lightning ================
    sdxl_pipeline = measure_time_and_usage("Load SDXL-Lightning", load_sdxl_lightning)
    throwaway_image = measure_time_and_usage("Generate (SDXL-Lightning)", generate_sdxl_lightning, sdxl_pipeline)
    measure_time_and_usage("Unload SDXL-Lightning", unload_model, sdxl_pipeline)

    # ================ 2) Instruct-Pix2Pix ================
    instruct_pix2pix = measure_time_and_usage("Load Instruct-Pix2Pix", load_instruct_pix2pix)
    edited_image = measure_time_and_usage("Generate (Instruct-Pix2Pix)", generate_instruct_pix2pix, instruct_pix2pix)
    measure_time_and_usage("Unload Instruct-Pix2Pix", unload_model, instruct_pix2pix)

    # ================ 3) Stable Video Diffusion ================
    video_diffusion = measure_time_and_usage("Load Stable Video Diffusion", load_video_diffusion)
    generated_video = measure_time_and_usage("Generate (Stable Video Diffusion)", generate_video_diffusion, video_diffusion)
    measure_time_and_usage("Unload Stable Video Diffusion", unload_model, video_diffusion)

    # Final usage measurement
    print_usage("End of Script")

    # Shutdown NVML
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()
