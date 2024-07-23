import base64
import io
import whisper
from flask import Flask, render_template, request, jsonify, send_file
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

app = Flask(__name__, template_folder=".")
socketio = SocketIO(app, async_mode="threading")

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
transcription_language = None
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
    print(f"Processing command: {command}")
    result = llm.invoke(Path('llm_instructions.txt').read_text().replace("<user_input>", command))
    return handle_llm_response(result.content)

def handle_llm_response(response):
    print(f"LLM response: {response}")
    response = json.loads(response)
    if "create" in response:
        prompt = response["create"]
        socketio.emit("status", "Creating new image...")
        return generate_image(prompt)
    if "edit" in response:
        prompt = response["edit"]
        socketio.emit("status", "Editing image...")
        return edit_image(prompt)
    if "video" in response:
        socketio.emit("status", "Generating video from image...")
        return generate_video_from_image()
    if "undo" in response:
        socketio.emit("status", "Reverting to previous image...")
        return previous_image()
    if "error" in response:
        return jsonify({"error": response["error"]})

def progress(pipe, step: int, timestep: int, callback_kwargs):
    print(f"Progress: Step {step}, Timestep {timestep}")
    socketio.emit("status", f"Generating, Step {step+1}")
    #latents = callback_kwargs["latents"]
    #image = latents_to_rgb(latents)
    #image.save(f"{step}.png")
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
    txt2img = load_sdxl_lightning()
    image = txt2img(
        prompt, 
        num_inference_steps=4, 
        guidance_scale=0,
        callback_on_step_end=progress,
        callback_on_step_end_tensor_inputs=["latents"]
    ).images[0]
    unload_model(txt2img)
    print("Image generated!")
    image.save("generated_image.webp")
    return send_file("generated_image.webp", mimetype="image/webp")

def edit_image(prompt):
    print(f"Editing image with prompt: {prompt}")
    image = Image.open("generated_image.webp")
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
        callback_on_step_end=progress
    ).images[0]
    unload_model(pix2pix)
    print("Edited image!")
    image = image.resize((resolution_hd, resolution_hd))
    image.save("edited_image.webp")
    return send_file("edited_image.webp", mimetype="image/webp")

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

def generate_video_from_image():
    print("Generating video from image...")
    image = Image.open("generated_image.webp")
    image = image.resize((1024, 576))
    img2vid = load_video_diffusion()
    frames = img2vid(
        image, 
        decode_chunk_size=2, 
        num_frames=7,
        num_inference_steps=10,
        callback_on_step_end=progress
    ).frames[0]
    print("Video generated!")
    unload_model(img2vid)
    export_to_video(frames, "generated_video.mp4", fps=7)
    mp4_to_webp("generated_video.mp4", "generated_video.webp", 7)
    return send_file("generated_video.webp", mimetype="image/webp")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)