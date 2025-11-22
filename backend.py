"""
Backend server for local inference - runs on powerful machine
Handles: Whisper transcription, LLM, Image generation/editing
"""
import base64
import io
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from queue import Queue
from threading import Lock, Thread
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import socketio
import torch
from transformers import T5EncoderModel, BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    EulerAncestralDiscreteScheduler,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    StableDiffusionXLInstructPix2PixPipeline,
    AutoencoderKL,
    EDMEulerScheduler,
    FluxTransformer2DModel,
)
from huggingface_hub import hf_hub_download, login
from safetensors.torch import load_file
from PIL import Image
from faster_whisper import WhisperModel, BatchedInferencePipeline
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment

# Load environment
load_dotenv()
CACHE_DIR = "./cache"
TEMP_AUDIO_DIR = "./temp_audio"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "hf.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF:Q4_K_M")
LOCAL_MODEL_SIZE = os.getenv("LOCAL_MODEL_SIZE", "turbo")

login(HUGGINGFACE_TOKEN)

app = FastAPI()

# Socket.IO for progress updates
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio, app)

# Models
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Backend using device: {device}")

# Queue system for image generation
image_queue = Queue()
image_results: Dict[str, Any] = {}  # task_id -> task info
image_queue_lock = Lock()
task_to_session: Dict[str, str] = {}  # task_id -> session_id for socket emissions

# Locks for different operations
whisper_lock = Lock()
llm_lock = Lock()

# Model placeholders
whisper_model = None
sdxl_l_txt2img = None
pix2pix_img2img = None
sd_cosxl_img2img = None
ollama_client = None

# Pydantic models
class LLMOutput(BaseModel):
    action: str
    prompt: str

class TranscribeRequest(BaseModel):
    audio_base64: str
    language: Optional[str] = None

class LLMRequest(BaseModel):
    user_prompt: str

class ImageGenerateRequest(BaseModel):
    prompt: str
    session_id: str

class ImageEditRequest(BaseModel):
    prompt: str
    image_base64: str
    session_id: str
    model: str = "fast"

class ImageQueueStatus(BaseModel):
    task_id: str
    position: int
    status: str

# Load models
def load_whisper_model():
    global whisper_model
    print("Loading WHISPER model...")
    model = WhisperModel(LOCAL_MODEL_SIZE, device="cuda", compute_type="float16", download_root=CACHE_DIR)
    whisper_model = BatchedInferencePipeline(model=model)
    print("WHISPER model loaded!")

def load_sdxl_lightning():
    global sdxl_l_txt2img
    print("Loading SDXL-Lightning model...")
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
    sdxl_l_txt2img = txt2img
    print("SDXL-Lightning loaded!")

def load_instruct_pix2pix():
    global pix2pix_img2img
    print("Loading Instruct-Pix2Pix model...")
    pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR
    )
    pix2pix.to("cuda")
    pix2pix.scheduler = EulerAncestralDiscreteScheduler.from_config(pix2pix.scheduler.config)
    pix2pix_img2img = pix2pix
    print("Instruct-Pix2Pix loaded!")

def load_cosxl_edit():
    global sd_cosxl_img2img
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
    sd_cosxl_img2img = cosxl
    print("COSXL-Edit loaded!")

def load_ollama_llm():
    global ollama_client
    print("Loading Ollama LLM...")
    ollama_client = OpenAI(base_url=OLLAMA_URL, api_key="ollama")
    # Test it
    try:
        messages = [
            {"role": "system", "content": "You are a loader! You say 'I am loaded'."},
            {"role": "user", "content": "Tell me you are loaded!"}
        ]
        response = ollama_client.beta.chat.completions.parse(
            model=OLLAMA_MODEL,
            messages=messages,
            max_tokens=100,
            response_format=LLMOutput,
            extra_body={"num_ctx": 1280}
        )
        print("Ollama LLM loaded!")
    except Exception as e:
        print(f"Error loading Ollama: {e}")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    load_ollama_llm()
    load_whisper_model()
    load_sdxl_lightning()
    load_instruct_pix2pix()
    load_cosxl_edit()
    # Start image queue processor
    Thread(target=process_image_queue, daemon=True).start()

# Health check
@app.get("/health")
async def health_check():
    """Check if backend is available and ready"""
    return {"status": "ok", "device": device}

# Transcription endpoint
@app.post("/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """Transcribe audio using local Whisper"""
    temp_audio = None
    temp_wav = None

    try:
        # Decode base64 audio
        audio_data = base64.b64decode(request.audio_base64)

        # Save temporarily in temp_audio directory
        # Try to detect format - if it's already WAV, use it directly
        temp_audio = os.path.join(TEMP_AUDIO_DIR, f"temp_audio_{uuid.uuid4()}.wav")

        with open(temp_audio, "wb") as f:
            f.write(audio_data)

        # Try to load the audio - it might be WAV or webm
        try:
            audio = AudioSegment.from_file(temp_audio, format="wav")
        except:
            # If WAV fails, try webm
            os.remove(temp_audio)
            temp_audio = os.path.join(TEMP_AUDIO_DIR, f"temp_audio_{uuid.uuid4()}.webm")
            with open(temp_audio, "wb") as f:
                f.write(audio_data)
            audio = AudioSegment.from_file(temp_audio, format="webm")

        # Check audio length
        if len(audio) < 500:  # Reduced from 2000ms to 500ms for partial transcriptions
            return {"transcription": "", "error": "Audio too short"}

        # Convert to wav if needed
        if not temp_audio.endswith('.wav'):
            temp_wav = os.path.join(TEMP_AUDIO_DIR, f"temp_audio_{uuid.uuid4()}.wav")
            audio.export(temp_wav, format="wav")
        else:
            temp_wav = temp_audio

        # Transcribe with lock
        with whisper_lock:
            segments, _ = whisper_model.transcribe(
                temp_wav,
                language=request.language,
                task="transcribe",
                batch_size=16
            )
            result_text = ' '.join([segment.text for segment in segments])

        print(f"Partial transcription result: '{result_text}'")
        return {"transcription": result_text}

    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp files
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception as e:
                print(f"Error removing temp audio: {e}")
        if temp_wav and temp_wav != temp_audio and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except Exception as e:
                print(f"Error removing temp wav: {e}")

# LLM endpoint
@app.post("/llm")
async def process_llm(request: LLMRequest):
    """Process LLM request to determine action and prompt"""
    try:
        system_prompt = Path('intention_recognition_prompt_v3_no_video.txt').read_text()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_prompt}
        ]

        # Can run multiple LLM requests concurrently but with rate limiting
        with llm_lock:
            response = ollama_client.beta.chat.completions.parse(
                model=OLLAMA_MODEL,
                messages=messages,
                max_tokens=200,
                response_format=LLMOutput,
                extra_body={"num_ctx": 1280}
            )

        result = response.choices[0].message.parsed
        return {"action": result.action, "prompt": result.prompt}

    except Exception as e:
        print(f"LLM error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Image queue processor
def process_image_queue():
    """Process image generation/editing tasks one at a time"""
    while True:
        # Update queue positions for all waiting tasks
        update_queue_positions()

        task = image_queue.get()
        task_id = task["task_id"]
        task_type = task["type"]
        session_id = task.get("session_id")

        try:
            # Update status to processing
            with image_queue_lock:
                image_results[task_id]["status"] = "processing"
                image_results[task_id]["position"] = 0

            # Emit processing started event
            if session_id:
                asyncio.run(sio.emit('image_processing_started', {
                    'task_id': task_id
                }, room=session_id))

            if task_type == "generate":
                result = generate_image_internal(
                    task["prompt"],
                    task["num_steps"],
                    task["guidance_scale"],
                    task_id,
                    session_id
                )
            elif task_type == "edit":
                result = edit_image_internal(
                    task["prompt"],
                    task["image"],
                    task["model"],
                    task_id,
                    session_id
                )

            # Store result
            with image_queue_lock:
                image_results[task_id]["status"] = "completed"
                image_results[task_id]["result"] = result

            # Emit completion event
            if session_id:
                asyncio.run(sio.emit('image_processing_completed', {
                    'task_id': task_id,
                    'result': result
                }, room=session_id))

        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            with image_queue_lock:
                image_results[task_id]["status"] = "error"
                image_results[task_id]["error"] = str(e)

            # Emit error event
            if session_id:
                asyncio.run(sio.emit('image_processing_error', {
                    'task_id': task_id,
                    'error': str(e)
                }, room=session_id))

        finally:
            image_queue.task_done()

def update_queue_positions():
    """Update queue positions for all waiting tasks and emit updates"""
    with image_queue_lock:
        # Get all queued tasks
        queued_tasks = [(tid, info) for tid, info in image_results.items()
                       if info["status"] == "queued"]

        # Sort by creation order (assuming task_ids are ordered)
        queued_tasks.sort(key=lambda x: x[0])

        # Update positions
        for position, (task_id, info) in enumerate(queued_tasks):
            old_position = info.get("position", -1)
            new_position = position

            if old_position != new_position:
                info["position"] = new_position
                session_id = task_to_session.get(task_id)

                if session_id:
                    asyncio.run(sio.emit('queue_position_update', {
                        'task_id': task_id,
                        'position': new_position
                    }, room=session_id))

def generate_image_internal(prompt: str, num_steps: int, guidance_scale: float, task_id: str, session_id: Optional[str] = None):
    """Internal image generation with progress updates"""
    def progress_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
        # Update progress
        progress_data = {
            "step": step + 1,
            "total_steps": num_steps
        }

        with image_queue_lock:
            image_results[task_id]["progress"] = progress_data

        # Emit progress update via socket
        if session_id:
            asyncio.run(sio.emit('image_progress', {
                'task_id': task_id,
                'progress': progress_data
            }, room=session_id))

        # Generate preview every few steps
        if step % 2 == 0:
            latents = callback_kwargs.get("latents")
            if latents is not None:
                with torch.no_grad():
                    latents_scaled = (1 / 0.18215) * latents
                    weights = ((60, -60, 25, -70), (60, -5, 15, -50), (60, 10, -5, -35))
                    weights_tensor = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
                    biases_tensor = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
                    rgb_tensor = torch.einsum("...lxy,lr -> ...rxy", latents_scaled, weights_tensor) + biases_tensor.unsqueeze(-1).unsqueeze(-1)
                    image_array = rgb_tensor.clamp(0, 255)[0].byte().cpu().numpy()
                    image_array = image_array.transpose(1, 2, 0)
                    preview_image = Image.fromarray(image_array)

                    # Store preview
                    buffered = io.BytesIO()
                    preview_image.save(buffered, format="PNG")
                    preview_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    with image_queue_lock:
                        image_results[task_id]["preview"] = preview_base64

                    # Emit preview via socket
                    if session_id:
                        asyncio.run(sio.emit('image_preview', {
                            'task_id': task_id,
                            'preview': preview_base64
                        }, room=session_id))

        return callback_kwargs

    image = sdxl_l_txt2img(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        callback_on_step_end=progress_callback
    ).images[0]

    # Convert to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {"image_base64": image_base64}

def edit_image_internal(prompt: str, image: Image.Image, model: str, task_id: str, session_id: Optional[str] = None):
    """Internal image editing with progress updates"""
    def latents_to_rgb(latents):
        """Convert latents to RGB for preview"""
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

    def progress_callback(pipe, step: int, timestep: int, callback_kwargs: dict):
        progress_data = None
        if model == "fast":
            progress_data = {"step": step + 1, "total_steps": 40}
        else:
            progress_data = {"step": step + 1, "total_steps": 20}

        with image_queue_lock:
            image_results[task_id]["progress"] = progress_data

        # Emit progress update via socket
        if session_id:
            asyncio.run(sio.emit('image_progress', {
                'task_id': task_id,
                'progress': progress_data
            }, room=session_id))

        # Generate preview images
        latents = callback_kwargs.get("latents")
        if latents is not None:
            generate_preview = False

            if model == "fast" and step % 2 == 0:  # Every 2 steps for fast model
                generate_preview = True
            elif model == "slow" and step % 2 == 0:  # Every 2 steps for slow model
                generate_preview = True

            if generate_preview:
                try:
                    if model == "fast":
                        # Use latents_to_rgb for fast model
                        preview_image = latents_to_rgb(latents)
                    else:
                        # Use VAE decode for slow model
                        with torch.no_grad():
                            latents_scaled = (1 / 0.18215) * latents
                            image_tensor = sd_cosxl_img2img.vae.decode(latents_scaled).sample
                            image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                            image_array = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                            pil_images = sd_cosxl_img2img.numpy_to_pil(image_array)
                            preview_image = pil_images[0]

                    buffered = io.BytesIO()
                    preview_image.save(buffered, format="PNG")
                    preview_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    with image_queue_lock:
                        image_results[task_id]["preview"] = preview_base64

                    # Emit preview via socket
                    if session_id:
                        asyncio.run(sio.emit('image_preview', {
                            'task_id': task_id,
                            'preview': preview_base64
                        }, room=session_id))
                except Exception as e:
                    print(f"Preview generation error: {e}")

        return callback_kwargs

    def progress_callback_slow(step: int, timestep: int, latents: torch.Tensor):
        """Callback for slow model (cosxl) - uses different signature without callback_kwargs"""
        progress_data = {"step": step + 1, "total_steps": 20}

        with image_queue_lock:
            image_results[task_id]["progress"] = progress_data

        # Emit progress update via socket
        if session_id:
            asyncio.run(sio.emit('image_progress', {
                'task_id': task_id,
                'progress': progress_data
            }, room=session_id))

        # Generate preview every 22 steps
        if step % 22 == 0:
            try:
                with torch.no_grad():
                    latents_scaled = (1 / 0.18215) * latents
                    image_tensor = sd_cosxl_img2img.vae.decode(latents_scaled).sample
                    image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
                    image_array = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                    pil_images = sd_cosxl_img2img.numpy_to_pil(image_array)
                    preview_image = pil_images[0]

                    buffered = io.BytesIO()
                    preview_image.save(buffered, format="PNG")
                    preview_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    with image_queue_lock:
                        image_results[task_id]["preview"] = preview_base64

                    # Emit preview via socket
                    if session_id:
                        asyncio.run(sio.emit('image_preview', {
                            'task_id': task_id,
                            'preview': preview_base64
                        }, room=session_id))
            except Exception as e:
                print(f"Preview generation error: {e}")

    if model == "fast":
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        edited_image = pix2pix_img2img(
            prompt,
            image=image,
            num_inference_steps=40,
            callback_on_step_end=progress_callback
        ).images[0]
        edited_image = edited_image.resize((1024, 1024), Image.Resampling.LANCZOS)
    else:  # slow
        edited_image = sd_cosxl_img2img(
            prompt=prompt,
            image=image,
            num_inference_steps=20,
            callback=progress_callback_slow,
            callback_steps=1
        ).images[0]

    # Convert to base64
    buffered = io.BytesIO()
    edited_image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return {"image_base64": image_base64}

# Image generation endpoint
@app.post("/generate_image")
async def generate_image(request: ImageGenerateRequest):
    """Queue image generation task"""
    task_id = str(uuid.uuid4())
    session_id = request.session_id

    # Add to queue
    with image_queue_lock:
        position = image_queue.qsize() + 1
        image_results[task_id] = {
            "status": "queued",
            "position": position,
            "progress": None,
            "preview": None,
            "result": None,
            "error": None
        }
        task_to_session[task_id] = session_id

    image_queue.put({
        "task_id": task_id,
        "type": "generate",
        "prompt": request.prompt,
        "num_steps": 4,
        "guidance_scale": 0,
        "session_id": session_id
    })

    # Emit initial queue position
    await sio.emit('queue_position_update', {
        'task_id': task_id,
        'position': position
    }, room=session_id)

    return {"task_id": task_id, "position": position}

# Image editing endpoint
@app.post("/edit_image")
async def edit_image(request: ImageEditRequest):
    """Queue image editing task"""
    task_id = str(uuid.uuid4())
    session_id = request.session_id

    # Decode image
    image_data = base64.b64decode(request.image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Add to queue
    with image_queue_lock:
        position = image_queue.qsize() + 1
        image_results[task_id] = {
            "status": "queued",
            "position": position,
            "progress": None,
            "preview": None,
            "result": None,
            "error": None
        }
        task_to_session[task_id] = session_id

    image_queue.put({
        "task_id": task_id,
        "type": "edit",
        "prompt": request.prompt,
        "image": image,
        "model": request.model,
        "session_id": session_id
    })

    # Emit initial queue position
    await sio.emit('queue_position_update', {
        'task_id': task_id,
        'position': position
    }, room=session_id)

    return {"task_id": task_id, "position": position}

# Task status endpoint
@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a queued/processing task"""
    with image_queue_lock:
        if task_id not in image_results:
            raise HTTPException(status_code=404, detail="Task not found")

        task_info = image_results[task_id]
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "position": task_info.get("position", 0),
            "progress": task_info.get("progress"),
            "preview": task_info.get("preview"),
            "result": task_info.get("result"),
            "error": task_info.get("error")
        }

# Socket.IO events
@sio.event
async def connect(sid, environ):
    """Handle client connection to backend"""
    print(f'Backend client connected: {sid}')

@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f'Backend client disconnected: {sid}')

@sio.event
async def join_session(sid, data):
    """Join a session room for receiving updates"""
    session_id = data.get('session_id')
    if session_id:
        await sio.enter_room(sid, session_id)
        print(f'Client {sid} joined session room: {session_id}')
        await sio.emit('session_joined', {'session_id': session_id}, room=sid)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)
