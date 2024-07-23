import base64
import io
import whisper
from flask import Flask, render_template, request, jsonify
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

app = Flask(__name__, template_folder=".")
socketio = SocketIO(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

print("Loading LLM...")
llm = ChatOllama(model="mistral:instruct")
print(llm.invoke("Respond with: Mistral-instruct ready to server!").content)

print("Loading WHISPER model...")
whisper_model = whisper.load_model("medium").to(device)
print("WHISPER model loaded successfully!")

# Holder for whole recording
audio_segments = []
transcription_language = ""
cache_dir = "./model_cache"

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
    img2vid = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16", cache_dir=cache_dir)
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
    print(f"Processing command: {command}")
    result = llm.invoke(Path('llm_instructions.txt').read_text().replace("<user_input>", command))
    return handle_llm_response(result.content)

def handle_llm_response(response):
    print(f"LLM response: {response}")
    response = json.loads(response)
    if "create" in response:
        prompt = response["create"]
        #socketio.emit("status", "Creating new image...")
        return generate_image(prompt)
    if "edit" in response:
        prompt = response["edit"]
        #socketio.emit("status", "Editing image...")
        return edit_image(prompt)
    if "video" in response:
        #socketio.emit("status", "Generating video from image...")
        return generate_video_from_image()
    if "undo" in response:
        #socketio.emit("status", "Reverting to previous image...")
        return previous_image()
    if "error" in response:
        return jsonify({"error": response["error"]})

def generate_image(prompt):
    print(f"Generating image for prompt: {prompt}")
    txt2img = load_sdxl_lightning()
    image = txt2img(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    unload_model(txt2img)
    print("Image generated!")
    image.save("generated_image.webp")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

def edit_image(prompt):
    print(f"Editing image with prompt: {prompt}")
    image = Image.open("generated_image.png")
    resolution = 512
    resolution_hd = 1024
    image = image.resize((resolution, resolution))
    pix2pix = load_instruct_pix2pix()
    image = pix2pix(
        prompt=prompt,
        image=image,
        guidance_scale=3,
        image_guidance_scale=1,
        num_inference_steps=10,
    ).images[0]
    unload_model(pix2pix)
    print("Edited image!")
    image = image.resize((resolution_hd, resolution_hd))
    image.save("edited_image.png")
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

def previous_image():
    print("Going to previous image")
    init_image = Image.open("generated_image.png")
    buffered = io.BytesIO()
    init_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

def generate_video_from_image():
    print("Generating video from image...")
    image = Image.open("generated_image.png")
    image = image.resize((1024, 576))
    generator = torch.manual_seed(42)
    img2vid = load_video_diffusion()
    frames = img2vid(image, decode_chunk_size=4, generator=generator, num_frames=7).frames[0]
    print("Video generated!")
    unload_model(img2vid)
    export_to_video(frames, "generated.mp4", fps=7)
    return jsonify({"video": "generated.mp4"})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)