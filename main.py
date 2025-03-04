import base64
import gc
import io
import os
import random
import shutil
import string
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import torch
from transformers import T5EncoderModel, pipeline, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import ( 
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    EulerAncestralDiscreteScheduler, 
    StableDiffusionInstructPix2PixPipeline, 
    StableVideoDiffusionPipeline, 
    StableDiffusionXLPipeline, 
    UNet2DConditionModel, 
    EulerDiscreteScheduler,
    StableDiffusionXLInstructPix2PixPipeline,
    AutoencoderKL,
    EDMEulerScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
    StableDiffusionLatentUpscalePipeline
)
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
from PIL import Image
from pathlib import Path
import json
from moviepy import *
from flask_cors import CORS
from threading import Lock
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
login(HUGGINGFACE_TOKEN)

# Set transcription method: "local" for local (faster-whisper) or "rest" for OpenAI Whisper REST API (requires API key)
TRANSCRIPTION_METHOD="local"
LOCAL_MODEL_SIZE="large-v3"

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
OPENAI_MODEL = "gpt-4o-mini"

# Setup LLM client based on provider choice.
def get_llm_client(selected_model) -> OpenAI: 
    if "openai" in selected_model.lower():
        return OpenAI(api_key=OPENAI_API_KEY)
    elif "local" in selected_model.lower():
        return OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    else:
        raise ValueError("Unsupported LLM Provider")

def get_llm_model(selected_model):
    if "openai" in selected_model.lower():
        return OPENAI_MODEL
    elif "local" in selected_model.lower():
        return OLLAMA_MODEL
    else:
        raise ValueError("Unsupported LLM Provider")

app = Flask(__name__, template_folder=".")
CORS(app)
socketio = SocketIO(app, async_mode="threading")

lock = Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

def load_whisper_model():
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", download_root=CACHE_DIR)
    return BatchedInferencePipeline(model=model)

def load_translator():
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fi-en", device=0)
    return translator

# https://huggingface.co/ByteDance/SDXL-Lightning
def load_sdxl_lightning():
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"
    unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
    unet = UNet2DConditionModel.from_config(unet_config).to("cuda", torch.float16)
    model_path = hf_hub_download(repo, ckpt, cache_dir=CACHE_DIR)
    unet.load_state_dict(load_file(model_path, device="cuda"))
    txt2img = StableDiffusionXLPipeline.from_pretrained(
        base, 
        unet=unet, 
        torch_dtype=torch.float16, 
        variant="fp16",
        cache_dir=CACHE_DIR
    )
    txt2img.to("cuda")
    txt2img.scheduler = EulerDiscreteScheduler.from_config(
        txt2img.scheduler.config, 
        timestep_spacing="trailing"
    )
    txt2img.enable_model_cpu_offload()
    return txt2img

# https://huggingface.co/black-forest-labs/FLUX.1-schnell
def load_flux1_schnell():
    quant_config = TransformersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    text_encoder_2_4bit = T5EncoderModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="text_encoder_2",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    quant_config = DiffusersBitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    transformer_4bit = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        subfolder="transformer",
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    flux1 = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        text_encoder_2=text_encoder_2_4bit,
        transformer=transformer_4bit,
        torch_dtype=torch.float16, 
        cache_dir=CACHE_DIR,
        low_cpu_mem_usage=True
    )
    flux1.enable_model_cpu_offload()
    return flux1

# https://huggingface.co/timbrooks/instruct-pix2pix
def load_instruct_pix2pix():
    pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix", 
        torch_dtype=torch.float16, 
        safety_checker=None,
        cache_dir=CACHE_DIR
    )
    pix2pix.to("cuda")
    pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(pix2pix.scheduler.config)
    pix2pix.enable_model_cpu_offload()
    return pix2pix

# https://huggingface.co/stabilityai/sd-x2-latent-upscaler
def load_sd_x2_lups():
    upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
        "stabilityai/sd-x2-latent-upscaler", 
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    upscaler.to("cuda")
    upscaler.enable_model_cpu_offload()
    return upscaler

# https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
# https://huggingface.co/stabilityai/cosxl
def load_cosxl_edit():
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", 
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    model_path = hf_hub_download(
        repo_id="stabilityai/cosxl",
        filename="cosxl_edit.safetensors",
        cache_dir=CACHE_DIR
    )
    cosxl = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
        model_path, 
        vae=vae, 
        torch_dtype=torch.float16, 
        num_in_channels=8, 
        is_cosxl_edit=True,
        cache_dir=CACHE_DIR
    )
    cosxl.to("cuda")
    cosxl.scheduler = EDMEulerScheduler(
        sigma_min=0.002, 
        sigma_max=120.0, 
        sigma_data=1.0, 
        prediction_type="v_prediction", 
        sigma_schedule="exponential"
    )
    cosxl.enable_model_cpu_offload()
    return cosxl

# https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
def load_video_diffusion():
    img2vid = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=CACHE_DIR
    )
    img2vid.to("cuda")
    img2vid.enable_model_cpu_offload()
    img2vid.unet.enable_forward_chunking()
    return img2vid

print("Loading models...")

print("Loading LLM...")
messages = [
    {"role": "system", "content": "You are a loader!"},
    {"role": "user", "content": "Tell me you are loaded!"}
]
response = get_llm_client("local").chat.completions.create(
    model=get_llm_model("local"),
    messages=messages,
    max_tokens=20,
)
print("LLM loaded successfully: " + response.choices[0].message.content)

print("Loading WHISPER model...")
whisper_model = load_whisper_model()
print("WHISPER model loaded successfully!")

print("Loading translator...")
translator = load_translator()
print("Translator loaded successfully!")

print("Loading FLUX.1-Schnell model...")
flux1_txt2img = load_flux1_schnell()
print("FLUX.1-Schnell model loaded successfully!")

print("Loading SDXL-Lightning model...")
sdxl_l_txt2img = load_sdxl_lightning()
print("SDXL-Lightning model loaded successfully!")

print("Loading Instruct-Pix2Pix model...")
pix2pix_img2img = load_instruct_pix2pix()
print("Instruct-Pix2Pix model loaded successfully!")

print("Loading SD-X2-Latent-Upscaler model...")
sd_x2_lups = load_sd_x2_lups()
print("SD-X2-Latent-Upscaler model loaded successfully!")

print("Loading Video-Diffusion model...")
svd_xt_img2vid = load_video_diffusion()
print("Video-Diffusion model loaded successfully!")

print("Loading COSXL-Edit model...")
sd_cosxl_img2img = load_cosxl_edit()
print("COSXL-Edit model loaded successfully!")

# empty torch cuda cache
torch.cuda.empty_cache()
gc.collect() 

# Holder for whole recording
audio_segments = []
transcription_language = None
selected_model = "fast_local"

def try_catch(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as e:
        print(f"An error occurred in {function}: {e}")
        socketio.emit("error", f"error: {e}")

class LLMOutput(BaseModel):
    action: str
    prompt: str

def poll_llm(user_prompt):
    system_prompt = Path('intention_recognition_prompt.txt').read_text()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response = get_llm_client(selected_model).beta.chat.completions.parse(
            model=get_llm_model(selected_model),
            messages=messages,
            max_tokens=200,
            response_format=LLMOutput
        )
        result = response.choices[0].message.parsed
        return result.action, result.prompt
    except Exception as e:
        print("Error in poll_llm:", e)
    return None, None

def run_whisper(audio_path, task="transcribe", language=None):
    if "local" in selected_model:
        with lock:
            # run local whisper
            print("Running local whisper...")
            if task == "translate" and language == "fi":
                segments, _ = whisper_model.transcribe(audio_path, language=language, task="transcribe", batch_size=16)
                result_text = ' '.join([segment.text for segment in segments])
                result_text = translator(result_text, max_length=512)[0]['translation_text']
            else:
                segments, _ = whisper_model.transcribe(audio_path, language=language, task=task, batch_size=16)
                result_text = ' '.join([segment.text for segment in segments])
            return result_text
    if "openai" in selected_model:
        # run openai whisper
        print("Running openai whisper...")
        if task == "transcribe":
            response = get_llm_client(selected_model).audio.transcriptions.create(model="whisper-1", file=audio_path, language=language)
        elif task == "translate":
            response = get_llm_client(selected_model).audio.translations.create(model="whisper-1", file=audio_path, language=language)
        else:
            return ""
        return response.text.strip()

    # error
    print("No whisper model selected!")
    return ""

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

@socketio.on("mdl_select")
def handle_lang_select(data):
    print(f"Selected models: {data}")
    global selected_model
    selected_model = data
    emit("status", "Updated model selection!")

@socketio.on("full_audio_data")
def handle_audio_data(data):
    print("Transcribing full audio data...")
    try_catch(process_full_audio, data)

def process_full_audio(data):
    decode = base64.b64decode(data)
    with open(f"full_audio.webm", "wb") as f:
        f.write(decode)
        f.close()
    audio = AudioSegment.from_file("full_audio.webm")
    if len(audio) < 2000:
        emit("empty_transcription", "No audio detected, please try again.")
        emit("status", "Waiting...")
        return
    result_text = run_whisper("full_audio.webm", "transcribe", transcription_language)
    if result_text == "":
        emit("empty_transcription", "No audio detected, please try again.")
        emit("status", "Waiting...")
        return
    print(f"Full transcription: {result_text}")
    emit("full_transcription", result_text)
    result_text = run_whisper("full_audio.webm", "translate", transcription_language)
    if result_text == "":
        emit("empty_transcription", "No audio detected, please try again.")
        emit("status", "Waiting...")
        return
    print(f"Full translation: {result_text}")
    emit("translation", result_text)

@socketio.on("audio_data")
def handle_audio_data(data):
    print("Transcribing audio data...")
    try_catch(process_transcription, data)

def process_transcription(data):
    decode = base64.b64decode(data)
    segment = AudioSegment.from_file(io.BytesIO(decode), format="wav")
    audio_segments.append(segment)

    with open(f"audio.wav", "wb") as f:
        f.write(decode)
        f.close()

    result_text = run_whisper("audio.wav", "transcribe", transcription_language)

    print(f"Transcription: {result_text}")
    emit("transcription", result_text)

@socketio.on("translate")
def handle_translation():
    print("Translating audio...")
    try_catch(process_translation)

def process_translation():
    save_concatenated_audio()
    with lock:
        result_text = run_whisper("concatenated_audio.wav", "translate", transcription_language)
    print(f"Translation: {result_text}")
    emit("translation", result_text)

@app.route("/process_command", methods=["POST"])
def process_command():
    data = request.json
    command = data.get("command")
    image = data.get("image")
    print(f"Processing command: {command}")
    return try_catch(llm_process_command, image, command)

def llm_process_command(image, command):
    action, prompt = poll_llm(command)
    print(f"LLM response: {action}, {prompt}")
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
        return previous_image(image)
    if action == "unknown":
        return jsonify({"unknown": prompt})

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

    if "local" in selected_model:
        if "fast" in selected_model:
            image = sdxl_l_txt2img(
                prompt,
                num_inference_steps=4,
                guidance_scale=0,
                callback_on_step_end=progress
            ).images[0]
        if "slow" in selected_model:
            image = flux1_txt2img(
                prompt,
                num_inference_steps=4,
                guidance_scale=0,
                callback_on_step_end=progress
            ).images[0]
    if "openai" in selected_model:
        response = get_llm_client(selected_model).images.generate(
            prompt=prompt,
            model="dall-e-3",
            size="1024x1024",
            response_format="b64_json",
            quality="standard",
            n=1,
        )
        image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))

    image = save_image(image, prompt)
    return jsonify({"image": image, "prompt": prompt})

def edit_image(parent_image, prompt):
    print(f"Editing image with prompt: {prompt}")
    image = get_saved_image(parent_image)

    if "local" in selected_model:
        if "fast" in selected_model:
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
            low_res_latents = pix2pix_img2img(
                prompt, 
                image=image, 
                num_inference_steps=20,
                output_type="latent",
                callback_on_step_end=progress
            ).images
            image = sd_x2_lups(
                prompt=prompt,
                image=low_res_latents,
                num_inference_steps=20,
                guidance_scale=0,
                callback_on_step_end=progress
            ).images[0]
        if "slow" in selected_model:
            image = sd_cosxl_img2img(
                prompt=prompt,
                image=image,
                num_inference_steps=20,
                callback_on_step_end=progress
            ).images[0]
    if "openai" in selected_model:
        response = get_llm_client(selected_model).images.edit(
            prompt=prompt,
            image=image,
            mask=open("dalle2_mask.png", "rb"),
            model="dall-e-2",
            size="1024x1024",
            response_format="b64_json",
            quality="standard",
            n=1,
        )
        image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))

    print("Edited image!")
    image = save_image(image, prompt, parent=parent_image)
    return jsonify({"image": image, "prompt": prompt})

def previous_image(image):
    print("Going to previous image")
    gallery_json = json.load(open("gallery.json", "r"))
    for obj in gallery_json:
        if obj["name"] == image:
            prompt = obj["prompt"]
            parent = obj["parent"]
            break
    
    if parent:
        return jsonify({"image": parent, "prompt": prompt, "action": "undo"})
    
    image_file = get_previous_image("./gallery", image + ".webp")
    for obj in gallery_json:
        if obj["name"] == image_file:
            prompt = obj["prompt"]
            break
    image_file = image_file.replace(".webp", "")
    return jsonify({"image": image_file, "prompt": prompt, "action": "undo"})

def mp4_to_webp(mp4_path, webp_path, fps):
    clip = VideoFileClip(mp4_path)
    forward_clip = clip
    backward_clip = clip.with_effects([vfx.TimeMirror()])
    looping_clip = concatenate_videoclips([forward_clip, backward_clip])

    # Save frames as individual WebP images
    frames = []
    for frame in looping_clip.iter_frames(fps=fps):
        img = Image.fromarray(frame)
        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
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
    image = get_saved_image(parent_image)
    image = image.resize((1024, 576), Image.Resampling.LANCZOS)
    frames = svd_xt_img2vid(
        image, 
        decode_chunk_size=2, 
        num_inference_steps=10,
        callback_on_step_end=progress
    ).frames[0]
    print("Video generated!")
    export_to_video(frames, "generated_video.mp4", fps=7)
    mp4_to_webp("generated_video.mp4", "generated_video.webp", 7)
    image = Image.open("generated_video.webp")
    image = save_image(image, prompt, parent=parent_image)
    shutil.copyfile("generated_video.webp", "./gallery/" + image + ".webp")
    return jsonify({"image": image, "prompt": prompt})

def get_saved_image(image_name):
    path = f"./gallery/{image_name}.webp"
    file_size = os.path.getsize(path)
    if file_size < 500 * 1024:
        return Image.open(path)
    gallery_json = json.load(open("gallery.json", "r"))
    for obj in gallery_json:
        if obj["name"] == image_name:
            if obj["parent"]:
                return get_saved_image(obj["parent"])
    return None

@app.route("/gallery")
def get_gallery_json():
    return send_file("gallery.json", mimetype="application/json")

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
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
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

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)