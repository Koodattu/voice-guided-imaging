import base64
import io
import whisper
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pydub import AudioSegment
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
from PIL import Image
from pathlib import Path
from langchain_community.chat_models import ChatOllama
import json
from imaginairy.api.generate import imagine
from imaginairy.schema import ImaginePrompt, ControlInput, LazyLoadingImage, config
from imaginairy.api.video_sample import generate_video
from imaginairy import config

app = Flask(__name__, template_folder=".")
socketio = SocketIO(app)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device (cuda/cpu): {device}")

print("Loading LLM...")
llm = ChatOllama(model="mistral:instruct")
print(llm.invoke("Respond with: Mistral-instruct ready to server!").content)

print("Loading WHISPER model...")
whisper_model = whisper.load_model("small").to(device)
print("WHISPER model loaded successfully!")

# Load the image generation model
MODEL_WEIGHT_CONFIGS = [
    config.ModelWeightsConfig(
        name="sdxl_lightning_4step_unet",
        aliases=["sdxl_lightning_4step_unet"],
        architecture=config.MODEL_ARCHITECTURE_LOOKUP["sdxl"],
        defaults={
            "negative_prompt": config.DEFAULT_NEGATIVE_PROMPT,
            "composition_strength": 0.6,
        },
        weights_location="https://huggingface.co/ByteDance/SDXL-Lightning/resolve/c9a24f48e1c025556787b0c58dd67a091ece2e44/sdxl_lightning_4step.safetensors?download=true",
        #weights_location="https://huggingface.co/ByteDance/SDXL-Lightning/resolve/c9a24f48e1c025556787b0c58dd67a091ece2e44/sdxl_lightning_2step_unet.safetensors?download=true",
        #weights_location="https://huggingface.co/ByteDance/SDXL-Lightning/tree/c9a24f48e1c025556787b0c58dd67a091ece2e44"
    ),
    config.ModelWeightsConfig(
        name="sdxl-instructpix2pix-768",
        aliases=["sdxl-instructpix2pix-768"],
        architecture=config.MODEL_ARCHITECTURE_LOOKUP["sdxl"],
        defaults={
            "negative_prompt": config.DEFAULT_NEGATIVE_PROMPT,
            "composition_strength": 0.6,
        },
        weights_location="https://huggingface.co/diffusers/sdxl-instructpix2pix-768/tree/06653d47f8d22f2c2205a5884d6a24c5e76d2ca7"
    ),
]

# Adding the custom model to the lookup
for mw in MODEL_WEIGHT_CONFIGS:
    for a in mw.aliases:
        config.MODEL_WEIGHT_CONFIG_LOOKUP[a] = mw

sdxl_txt2img = config.MODEL_WEIGHT_CONFIG_LOOKUP["sdxl_lightning_4step_unet"]
sdxl_pix2pix = config.MODEL_WEIGHT_CONFIG_LOOKUP["sdxl-instructpix2pix-768"]

# Holder for whole recording
audio_segments = []
transcription_language = ""

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
    prompt = ImaginePrompt(prompt=prompt, model_weights="sd15", steps=10, size="1024x1024")
    result = next(imagine(prompts=prompt))
    print("Image generated!")
    result.img.save("generated_image.png")
    buffered = io.BytesIO()
    result.img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return jsonify({"image": img_str})

def edit_image(prompt):
    print(f"Editing image with prompt: {prompt}")
    image = Image.open("generated_image.png")
    image = image.resize((768, 768))
    image = LazyLoadingImage(img = image)
    control_mode = ControlInput(mode="edit", image=image)
    prompt = ImaginePrompt(size="1024x1024", prompt=prompt, control_inputs=[control_mode], init_image_strength=0.01, steps=10, model_weights="sd15")
    imagine_iterator = imagine(prompts=prompt)
    result = next(imagine_iterator)
    print("Edited image!")
    result.img.save("edited_image.png")
    buffered = io.BytesIO()
    result.img.save(buffered, format="PNG")
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
    video = generate_video(input_path="generated_image.png", output_folder="./")
    return jsonify({"video": video})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)