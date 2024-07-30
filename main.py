import base64
import io
import os
import random
import shutil
import string
import whisper
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import torch
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline, StableVideoDiffusionPipeline, StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler,StableDiffusionXLInstructPix2PixPipeline
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
from pathlib import Path
from langchain_community.chat_models import ChatOllama
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from flask_cors import CORS
import pandas as pd

app = Flask(__name__, template_folder=".")
CORS(app)
socketio = SocketIO(app, async_mode="threading")

cache_dir = "./model_cache"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

print("Loading LLM...")
llm = ChatOllama(model="mistral:instruct")
print(llm.invoke("Respond with: Mistral-instruct ready to server!").content)

print("Loading WHISPER model...")
whisper_model = whisper.load_model("medium").to(device)
print("WHISPER model loaded successfully!")

print("Loading SDXL-Lightning model...")
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"
unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
txt2img = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
txt2img.to("cuda")
txt2img.scheduler = EulerDiscreteScheduler.from_config(txt2img.scheduler.config, timestep_spacing="trailing")
txt2img.enable_model_cpu_offload()
print("SDXL-Lightning model loaded successfully!")

# Holder for whole recording
audio_segments = []
transcription_language = None

# https://huggingface.co/ByteDance/SDXL-Lightning
def load_sdxl_lightning():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_2step_unet.safetensors"
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
    txt2img = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
    txt2img.to("cuda")
    txt2img.scheduler = EulerDiscreteScheduler.from_config(txt2img.scheduler.config, timestep_spacing="trailing")
    txt2img.enable_model_cpu_offload()
    return txt2img

# https://huggingface.co/timbrooks/instruct-pix2pix
def load_instruct_pix2pix():
    pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, cache_dir=cache_dir)
    pix2pix.to("cuda")
    pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(pix2pix.scheduler.config)
    pix2pix.enable_model_cpu_offload()
    return pix2pix

# https://huggingface.co/diffusers/sdxl-instructpix2pix-768
def load_sdxl_instruct_pix2pix():
    pix2pix_sdxl = StableDiffusionXLInstructPix2PixPipeline.from_pretrained("diffusers/sdxl-instructpix2pix-768", torch_dtype=torch.float16, cache_dir=cache_dir)
    pix2pix_sdxl.to("cuda")
    pix2pix_sdxl.enable_model_cpu_offload()
    return pix2pix_sdxl

# https://huggingface.co/docs/diffusers/using-diffusers/svd
def load_video_diffusion():
    img2vid = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt-1-1", torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
    img2vid.to("cuda")
    img2vid.enable_model_cpu_offload()
    img2vid.unet.enable_forward_chunking()
    return img2vid

def unload_model(model):
    del model
    torch.cuda.empty_cache()

def save_concatenated_audio():
    concatenated = AudioSegment.empty()
    for segment in audio_segments:
        concatenated += segment
    concatenated.export("concatenated_audio.wav", format="wav")
    audio_segments.clear()

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on("lang_select")
def handle_lang_select(data):
    print(f"Selected language: {data}")
    global transcription_language
    transcription_language = None if data == "" else data
    emit("status", "Updated language selection!")

@socketio.on("audio_data")
def handle_audio_data(data):
    print("Transcribing audio data...")
    decode = base64.b64decode(data)
    segment = AudioSegment.from_file(io.BytesIO(decode), format="wav")
    audio_segments.append(segment)

    with open(f"audio.wav", "wb") as f:
        f.write(decode)
        f.close()
    result = whisper_model.transcribe("audio.wav", language=transcription_language, fp16=True)
    print(f"Transcription: {result['text']}")
    emit("transcription", result["text"])

@socketio.on("translate")
def handle_translation():
    print("Translating audio...")
    save_concatenated_audio()
    result = whisper_model.transcribe("concatenated_audio.wav", task="translate", language=transcription_language, fp16=True)
    print(f"Translation: {result['text']}")
    emit("translation", result["text"])

@app.route("/process_command", methods=["POST"])
def process_command():
    data = request.json
    command = data.get("command")
    image = data.get("image")
    print(f"Processing command: {command}")
    return llm_process_command(image, command)

def llm_process_command(image, command):
    result = llm.invoke(Path('llm_instructions_command.txt').read_text().replace("<user_input>", command))
    print(f"LLM response: {result.content}")
    response = json.loads(result.content)
    action = response["action"]
    prompt = response["prompt"]
    socketio.emit("llm_response", action + ": " + prompt)
    if action == "create":
        socketio.emit("status", "Creating new image...")
        return generate_image(prompt)
    if action == "edit":
        socketio.emit("status", "Editing image...")
        return edit_image(image, prompt)
    if action == "video":
        socketio.emit("status", "Generating video from image...")
        return generate_video_from_image(image, prompt)
    if action == "undo":
        socketio.emit("status", "Reverting to previous image...")
        return previous_image()
    if action == "error":
        return jsonify({"error": prompt})

def progress(pipe, step: int, timestep: int, callback_kwargs):
    socketio.emit("status", f"Generating, Step {step+1}")
    if "StableVideoDiffusion" in str(pipe):
        return callback_kwargs
    latents = callback_kwargs["latents"]
    image = latents_to_rgb(latents)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    socketio.emit("image_progress", img_str)
    return callback_kwargs

def latents_to_rgb(latents):
    weights = (
        (60, -60, 25, -70),
        (60,  -5, 15, -50),
        (60,  10, -5, -35)
    )
    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
    image_array = image_array.transpose(1, 2, 0)
    return Image.fromarray(image_array)

def generate_image(prompt):
    print(f"Generating image for prompt: {prompt}")
    image = txt2img(
        prompt,
        num_inference_steps=4, 
        guidance_scale=0,
        callback_on_step_end=progress
    ).images[0]
    image = save_image(image, prompt)
    return jsonify({"image": image, "prompt": prompt})

def edit_image(parent_image, prompt):
    print(f"Editing image with prompt: {prompt}")
    image = Image.open("./gallery/" + parent_image + ".webp")
    #image = image.resize((768, 768))
    pix2pix = load_instruct_pix2pix()
    image = pix2pix(
        prompt=prompt,
        image=image,
        num_inference_steps=10,
        callback_on_step_end=progress
    ).images[0]
    unload_model(pix2pix)
    print("Edited image!")
    #image = image.resize((1024, 1024))
    image = save_image(image, prompt, parent=parent_image)
    return jsonify({"image": image, "prompt": prompt})

def previous_image():
    print("Going to previous image")
    return send_file("generated_image.webp", mimetype="image/webp")

def mp4_to_webp(mp4_path, webp_path, fps):
    clip = VideoFileClip(mp4_path)
    forward_clip = clip
    backward_clip = clip.fx(vfx.time_mirror)
    looping_clip = concatenate_videoclips([forward_clip, backward_clip])

    # Save frames as individual WebP images
    frames = []
    for frame in looping_clip.iter_frames(fps=fps):
        img = Image.fromarray(frame)
        img = img.resize((1024, 1024))
        frames.append(img)

    # Save frames as a looping WebP animation
    frames[0].save(
        webp_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0
    )

def generate_video_from_image(parent_image, prompt):
    print("Generating video from image...")
    image = Image.open("./gallery/" + parent_image + ".webp")
    image = image.resize((1024, 576))
    img2vid = load_video_diffusion()
    frames = img2vid(
        image, 
        decode_chunk_size=2, 
        num_inference_steps=10,
        callback_on_step_end=progress
    ).frames[0]
    print("Video generated!")
    unload_model(img2vid)
    export_to_video(frames, "generated_video.mp4", fps=7)
    mp4_to_webp("generated_video.mp4", "generated_video.webp", 7)
    image = Image.open("generated_video.webp")
    image = save_image(image, prompt, parent=parent_image)
    shutil.copyfile("generated_video.webp", "./gallery/" + image + ".webp")
    return jsonify({"image": image, "prompt": prompt})

@app.route("/images/<image>")
def get_image(image):
    return send_from_directory("./gallery", image + ".webp")

@app.route("/images")
def images():
    gallery_json = "gallery.json"
    if not os.path.exists(gallery_json):
        return jsonify([])
    with open(gallery_json, 'r') as file:
        images = json.load(file)
    return jsonify(images)

def get_sorted_images_by_date(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
    return files

def get_previous_image(folder_path, file_name):
    files = get_sorted_images_by_date(folder_path)
    index = files.index(file_name)
    if index == 0:
        return None
    return files[index - 1]

def random_image_name(prompt, length=6):
    words = prompt.split()[:4]
    words = "-".join(words)
    random_name = ''.join(random.choices(string.ascii_letters, k=length))
    return words + "-" + random_name

def save_image(image, prompt, parent=None):
    if not os.path.exists("./gallery"):
        os.makedirs("./gallery")
    if not os.path.exists("./gallery/thumbnails"):
        os.makedirs("./gallery/thumbnails")
    image_name = random_image_name(prompt)
    image.save(f"./gallery/{image_name}.webp")
    image = image.resize((256, 256))
    image.save(f"./gallery/thumbnails/{image_name}.webp")
    add_to_json_file(image_name, prompt, parent)
    return image_name

def add_to_json_file(name, prompt, parent):
    filename = "gallery.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)
    with open(filename, 'r') as file:
        data = json.load(file)
    new_entry = {
        "name": name,
        "prompt": prompt,
        "parent": parent
    }
    data.append(new_entry)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/gallery")
def gallery():
    return render_template("gallery.html")

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)