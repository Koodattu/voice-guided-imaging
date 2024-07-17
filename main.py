import base64
import io
import whisper
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

app = Flask(__name__, template_folder=".")
socketio = SocketIO(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

print("Loading WHISPER model...")
whisper_model = whisper.load_model("medium").to(device)

# Load the image generation model
print("Loading SDXL model...")
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_2step_unet.safetensors"  # Use the correct ckpt for your step setting!

# Load model configuration
config = UNet2DConditionModel.load_config(base, subfolder="unet")
unet = UNet2DConditionModel.from_config(config).to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

audio_segments = []

def save_concatenated_audio():
    concatenated = AudioSegment.empty()
    for segment in audio_segments:
        concatenated += segment
    concatenated.export("concatenated_audio.wav", format="wav")
    audio_segments.clear()

@socketio.on("audio_data")
def handle_audio_data(data):
    decode = base64.b64decode(data)
    segment = AudioSegment.from_file(io.BytesIO(decode), format="wav")
    audio_segments.append(segment)

    with open(f"audio.wav", "wb") as f:
        f.write(decode)
        f.close()

    result = whisper_model.transcribe("audio.wav")
    emit("transcription", result["text"])

@socketio.on("translate")
def handle_translation():
    save_concatenated_audio()
    result = whisper_model.transcribe("concatenated_audio.wav", task="translate")
    emit("translation", result["text"])

@app.route("/generate_image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    print(f"Generating image for prompt: {prompt}")
    image = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

@app.route("/")
def index():
    return render_template("index.html")

    
if __name__ == "__main__":
    print("Starting server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)