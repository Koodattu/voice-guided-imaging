import base64
import io
import json
import os
import random
import shutil
import string
import tempfile
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import torch
from transformers import T5EncoderModel, pipeline, BitsAndBytesConfig as TransformersBitsAndBytesConfig, CLIPTextModelWithProjection, CLIPTokenizer
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
    StableDiffusionLatentUpscalePipeline,
    AutoPipelineForText2Image,
    FluxKontextPipeline,
    GGUFQuantizationConfig,
    ZImagePipeline
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
from google import genai
from google.genai import types
from ollama import Client as OllamaClient

class LLMOutput(BaseModel):
    action: str
    prompt: str

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
login(HUGGINGFACE_TOKEN)

# Set transcription method: "local" for local (faster-whisper) or "rest" for OpenAI Whisper REST API (requires API key)
LOCAL_MODEL_SIZE="turbo" # "small", "medium", "large-v3", "turbo"
CLOUD_PROVIDER = "google" # "openai" or "google"

OLLAMA_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q4_K_M"#"qwen3:4b"
OPENAI_MODEL = "gpt-4o-mini"

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
OLLAMA_CLIENT = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
GOOGLE_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

# Setup LLM client based on provider choice.
def get_llm_client(selected_model) -> OpenAI:
    if "cloud" in selected_model.lower():
        print("Using OpenAI API")
        return OPENAI_CLIENT
    elif "local" in selected_model.lower():
        print("Using Ollama API")
        return OLLAMA_CLIENT
    else:
        raise ValueError("Unsupported LLM Provider")

def get_llm_model(selected_model):
    if "cloud" in selected_model.lower():
        print("Using OpenAI model")
        return OPENAI_MODEL
    elif "local" in selected_model.lower():
        print("Using Ollama model")
        return OLLAMA_MODEL
    else:
        raise ValueError("Unsupported LLM Provider")

app = Flask(__name__, template_folder=".")
CORS(app)
socketio = SocketIO(app, async_mode="threading", path="/kuvagen/socket.io")

lock = Lock()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

def load_whisper_model():
    print("Loading WHISPER model...")
    model = WhisperModel(LOCAL_MODEL_SIZE, device="cuda", compute_type="float16", download_root=CACHE_DIR)
    print("WHISPER model loaded successfully!")
    return BatchedInferencePipeline(model=model)

# https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
def load_zimage_turbo():
    print("Loading Z-Image-Turbo model...")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        cache_dir=CACHE_DIR
    )
    pipe.to("cuda")
    #pipe.transformer.compile()
    #pipe.transformer.set_attention_backend("flash")
    #pipe.transformer.set_attention_backend("_flash_3")
    pipe.enable_model_cpu_offload()
    print("Z-Image-Turbo model loaded successfully!")
    return pipe

# https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
# https://huggingface.co/stabilityai/cosxl
def load_cosxl_edit():
    print("Loading COSXL-Edit model...")
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
    print("COSXL-Edit model loaded successfully!")
    return cosxl

def load_ollama_llm():
    print("Loading LLM...")
    messages = [
        {"role": "system", "content": "You are a loader! You say 'I am loaded'."},
        {"role": "user", "content": "Tell me you are loaded!"}
    ]
    try:
        response = OLLAMA_CLIENT.beta.chat.completions.parse(
            model=get_llm_model("local"),
            messages=messages,
            max_tokens=100,
            response_format=LLMOutput,
            extra_body={"num_ctx": 1280}
        )
        print("LLM loaded successfully!")
    except Exception as e:
        print(f"Error loading LLM: {e}")

def periodic_ollama_loader():
    while True:
        try:
            load_ollama_llm()
        except Exception as e:
            print(f"Error in periodic Ollama loader: {e}")
        time.sleep(60)

print("Loading models...")

load_ollama_llm()
whisper_model = load_whisper_model()

# Better quality but slower local models
zimage_turbo = load_zimage_turbo()
sd_cosxl_img2img = load_cosxl_edit()

# Holder for whole recording
audio_segments = []
transcription_language = None
selected_model = "localfast"

def try_catch(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as e:
        print(f"An error occurred in {function}: {e}")
        socketio.emit("error", f"error: {e}")

def poll_llm(user_prompt):
    system_prompt = Path('intention_recognition_prompt_v3_no_video.txt').read_text()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        extra_body = None
        if "local" in selected_model:
            extra_body = {"num_ctx": 1280}

        response = get_llm_client(selected_model).beta.chat.completions.parse(
            model=get_llm_model(selected_model),
            messages=messages,
            max_tokens=200,
            response_format=LLMOutput,
            extra_body=extra_body,
        )
        result = response.choices[0].message.parsed
        return result.action, result.prompt
    except Exception as e:
        print("Error in poll_llm:", e)
    return None, None

def run_whisper(audio_path, language=None):
    print("Running !" + selected_model + "! Whisper for language !" + str(language) + "!")
    with lock:
        if "local" in selected_model:
            segments, _ = whisper_model.transcribe(audio_path, language=language, task="transcribe", batch_size=16)
            result_text = ' '.join([segment.text for segment in segments])
            return result_text
        if "cloud" in selected_model:
            try:
                with open(audio_path, "rb") as audio_file:
                    client = get_llm_client(selected_model)
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        response_format="text"
                    )
                return response
            except Exception as e:
                print(f"An error occurred during OpenAI Whisper processing: {e}")
                return ""

        # error
        print("No whisper model selected!")
        return ""

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
def handle_full_audio_data(data):
    print("Transcribing full audio data...")
    try_catch(process_full_audio, data)

def process_full_audio(data):
    decode = base64.b64decode(data)
    with open(f"full_audio.webm", "wb") as f:
        f.write(decode)
    audio = AudioSegment.from_file("full_audio.webm")
    if len(audio) < 2000:
        emit("empty_transcription", "No audio detected, please try again.")
        emit("status", "Waiting...")
        return
    result_text = run_whisper("full_audio.webm", transcription_language)
    if result_text == "":
        emit("empty_transcription", "No audio detected, please try again.")
        emit("status", "Waiting...")
        return
    print(f"Full transcription: {result_text}")
    emit("final_transcription_result", result_text)

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

    result_text = run_whisper("audio.wav", transcription_language)

    print(f"Transcription: {result_text}")
    emit("transcription", result_text)

def save_concatenated_audio():
    print(f"Concatenating {len(audio_segments)} audio segments...")
    concatenated = AudioSegment.empty()
    for segment in audio_segments:
        concatenated += segment
    print(f"Exporting concatenated audio ({len(concatenated)}ms)...")
    concatenated.export("concatenated_audio.wav", format="wav")
    print("Audio segments cleared")
    audio_segments.clear()

@socketio.on("final_transcription")
def handle_final_transcription():
    print("Processing final transcription...")
    try_catch(process_final_transcription)

def process_final_transcription():
    print("Starting save_concatenated_audio...")
    save_concatenated_audio()
    print("save_concatenated_audio completed, starting transcription...")
    result_text = run_whisper("concatenated_audio.wav", transcription_language)
    print(f"Final transcription: {result_text}")
    emit("final_transcription_result", result_text)

@app.route("/kuvagen/process_command", methods=["POST"])
def process_command():
    data = request.json
    command = data.get("command")
    image = data.get("image")
    print(f"Processing command: {command}")
    start_time = time.time()
    result = try_catch(llm_process_command, image, command)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"Time elapsed: {elapsed_time:.2f} ms")
    return result

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
    if action == "undo":
        socketio.emit("status", "Reverting to previous image...")
        return previous_image(image)
    if action == "error":
        return jsonify({"error": prompt})

def make_optimised_callback(pipe, frequency: int = 11):
    def callback(step: int, timestep: int, latents: torch.Tensor):
        socketio.emit("status", f"Generating, Step {step+1}")
        if step % frequency == 0:
            with torch.no_grad():
                latents_scaled = (1 / 0.18215) * latents
                image_tensor = pipe.vae.decode(latents_scaled).sample
                image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                image_array = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                pil_images = pipe.numpy_to_pil(image_array)

                if pil_images and len(pil_images) > 0:
                    image_to_send = pil_images[0]
                    buffered = io.BytesIO()
                    image_to_send.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    socketio.emit("image_progress", img_str)
    return callback

def progress_callback_on_step_end(pipe, step: int, timestep: int, callback_kwargs: dict):
    socketio.emit("status", f"Generating, Step {step+1}")
    latents = callback_kwargs.get("latents")
    latents_scaled = (1 / 0.18215) * latents
    image_tensor = pipe.vae.decode(latents_scaled).sample
    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
    image_array = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    pil_images = pipe.numpy_to_pil(image_array)
    buffered = io.BytesIO()
    pil_images[0].save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    socketio.emit("image_progress", img_str)
    return callback_kwargs

def progress_callback_old(pipe, step: int, timestep: int, callback_kwargs):
    socketio.emit("status", f"Generating, Step {step+1}")
    latents = callback_kwargs["latents"]
    image = latents_to_rgb_old(latents)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    socketio.emit("image_progress", img_str)
    return callback_kwargs

def latents_to_rgb_old(latents):
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
        print("Using SDXL Lightning for fast model option")
        image = zimage_turbo(
            prompt,
            height=1024,
            width=1024,
            num_inference_steps=9,
            guidance_scale=0,
            #callback_on_step_end=progress_callback_old
        ).images[0]
    if "cloud" in selected_model:
        if "openai" in CLOUD_PROVIDER:
            response = get_llm_client(selected_model).images.generate(
                prompt=prompt,
                model="dall-e-3",
                size="1024x1024",
                response_format="b64_json",
                quality="standard",
                n=1,
            )
            image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        if "google" in CLOUD_PROVIDER:
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please generate the following image: " + prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            image = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    break
            if image is None:
                raise Exception("No edited image output in response.")

    image = save_image(image, prompt)
    return jsonify({"image": image, "prompt": prompt})

def edit_image(parent_image, prompt):
    print(f"Editing image with prompt: {prompt}")
    image = get_saved_image(parent_image)

    if "local" in selected_model:
        callback = make_optimised_callback(sd_cosxl_img2img, frequency=21)
        image = sd_cosxl_img2img(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            callback=callback,
            callback_steps=1
        ).images[0]
    if "cloud" in selected_model:
        if "openai" in CLOUD_PROVIDER:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, 'temp_image.png')
                image = image.convert('RGBA')
                image.save(temp_file_path, format='PNG')
                with open(temp_file_path, 'rb') as image_file, open("dalle2_mask.png", "rb") as mask_file:
                    response = get_llm_client(selected_model).images.edit(
                        prompt=prompt,
                        image=image_file,
                        mask=mask_file,
                        model="dall-e-2",
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        if "google" in CLOUD_PROVIDER:
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please edit the image: " + prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            image = None
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    break
            if image is None:
                raise Exception("No edited image output in response.")

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

@app.route("/kuvagen/gallery")
def get_gallery_json():
    return send_file("gallery.json", mimetype="application/json")

@app.route("/kuvagen/images/<image>")
def get_image(image):
    return send_from_directory("./gallery", image + ".webp")

@app.route("/kuvagen/images")
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

@app.route("/kuvagen/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print("Server started, ready to go!")
    threading.Thread(target=periodic_ollama_loader, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)