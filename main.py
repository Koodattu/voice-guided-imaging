"""
Main web server - lightweight relay to backend.py or cloud services
Handles: Session management, user limits, routing requests
"""
import base64
import io
import json
import os
import random
import string
import time
import hashlib
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from threading import Thread, Lock

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, session
from flask_socketio import SocketIO, emit, join_room
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import requests
from pydub import AudioSegment
from pydantic import BaseModel
import socketio as socketio_client
import tempfile

# Load environment
load_dotenv()

# Create temp directory for audio files
TEMP_AUDIO_DIR = "./temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "google")  # "openai" or "google"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# User limits
MAX_IMAGES_WITHOUT_LOGIN = int(os.getenv("MAX_IMAGES_WITHOUT_LOGIN", "10"))

# Clients
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
GOOGLE_CLIENT = genai.Client(api_key=GEMINI_API_KEY)

# Flask app
app = Flask(__name__, template_folder=".")
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)
socketio = SocketIO(app, async_mode="threading", path="/kuvagen/socket.io", cors_allowed_origins="*")

# Backend Socket.IO client for receiving progress updates
backend_sio = None
backend_connected = False
backend_connection_lock = Lock()

# Session storage (in production, use Redis or database)
user_sessions: Dict[str, Dict[str, Any]] = {}
backend_available = False
backend_status_lock = Lock()

# Pydantic model for structured LLM output
class LLMOutput(BaseModel):
    action: str
    prompt: str

def get_user_id():
    """Get or create user ID based on request body or session"""
    # Try to get user_id from request body (for POST requests)
    if request.is_json and request.json and 'user_id' in request.json:
        user_id = request.json['user_id']
        session['user_id'] = user_id
        return user_id

    # Fall back to session-based approach
    if 'user_id' not in session:
        # Generate user ID from IP and user agent
        ip = request.remote_addr
        user_agent = request.headers.get('User-Agent', '')
        raw_id = f"{ip}_{user_agent}"
        user_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]
        session['user_id'] = user_id

    return session['user_id']

def get_or_create_session(user_id: str) -> Dict[str, Any]:
    """Get or create user session data"""
    if user_id not in user_sessions:
        # Load gallery from gallery.json if it exists
        gallery = load_user_gallery(user_id)

        # Default to local slow model if backend is available, otherwise cloud
        with backend_status_lock:
            default_model = "localslow" if backend_available else "cloud"

        user_sessions[user_id] = {
            "user_id": user_id,
            "created_at": datetime.now(),
            "images_generated": len(gallery),  # Count existing images
            "transcription_language": None,
            "selected_model": default_model,
            "gallery": gallery,
            "current_image": None
        }
    return user_sessions[user_id]

def load_user_gallery(user_id: str) -> list:
    """Load user's gallery from gallery.json"""
    filename = "gallery.json"
    if not os.path.exists(filename):
        return []

    try:
        with open(filename, 'r') as file:
            all_data = json.load(file)

        # Filter for this user's images
        user_gallery = []
        for entry in all_data:
            if entry.get("user_id") == user_id:
                user_gallery.append({
                    "name": entry["name"],
                    "prompt": entry["prompt"],
                    "parent": entry.get("parent"),
                    "created_at": entry.get("created_at", datetime.now().isoformat())
                })

        return user_gallery
    except Exception as e:
        print(f"Error loading gallery for user {user_id}: {e}")
        return []

def check_user_limit(user_id: str) -> bool:
    """Check if user has exceeded image generation limit"""
    user_data = get_or_create_session(user_id)
    return user_data["images_generated"] < MAX_IMAGES_WITHOUT_LOGIN

def increment_user_count(user_id: str):
    """Increment user's image generation count"""
    user_data = get_or_create_session(user_id)
    user_data["images_generated"] += 1

def check_backend_availability():
    """Check if backend.py is available"""
    global backend_available
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        status = response.status_code == 200
        with backend_status_lock:
            backend_available = status
        print(f"Backend availability: {backend_available}")

        # Connect to backend socket if available
        if status:
            connect_to_backend_socket()
    except Exception as e:
        with backend_status_lock:
            backend_available = False
        print(f"Backend not available: {e}")
    return backend_available

def connect_to_backend_socket():
    """Connect to backend's Socket.IO server"""
    global backend_sio, backend_connected

    with backend_connection_lock:
        if backend_connected:
            return

        try:
            backend_sio = socketio_client.Client()

            @backend_sio.event
            def connect():
                global backend_connected
                backend_connected = True
                print('Connected to backend Socket.IO server')

            @backend_sio.event
            def disconnect():
                global backend_connected
                backend_connected = False
                print('Disconnected from backend Socket.IO server')

            @backend_sio.on('queue_position_update')
            def on_queue_position_update(data):
                """Relay queue position updates to frontend"""
                task_id = data.get('task_id')
                position = data.get('position')

                # Find session for this task
                session_id = find_session_for_task(task_id)
                if session_id:
                    socketio.emit('backend_queue_position', {
                        'task_id': task_id,
                        'position': position
                    }, room=session_id)

            @backend_sio.on('image_processing_started')
            def on_processing_started(data):
                """Relay processing started event to frontend"""
                task_id = data.get('task_id')
                session_id = find_session_for_task(task_id)
                if session_id:
                    socketio.emit('backend_processing_started', {
                        'task_id': task_id
                    }, room=session_id)

            @backend_sio.on('image_progress')
            def on_image_progress(data):
                """Relay progress updates to frontend"""
                task_id = data.get('task_id')
                progress = data.get('progress')
                session_id = find_session_for_task(task_id)
                if session_id:
                    socketio.emit('backend_progress', {
                        'task_id': task_id,
                        'progress': progress
                    }, room=session_id)

            @backend_sio.on('image_preview')
            def on_image_preview(data):
                """Relay preview images to frontend"""
                task_id = data.get('task_id')
                preview = data.get('preview')
                session_id = find_session_for_task(task_id)
                if session_id:
                    socketio.emit('backend_preview', {
                        'task_id': task_id,
                        'preview': preview
                    }, room=session_id)

            @backend_sio.on('image_processing_completed')
            def on_processing_completed(data):
                """Handle completion event - save image and notify frontend"""
                task_id = data.get('task_id')
                result = data.get('result')
                session_id = find_session_for_task(task_id)

                if session_id and result and 'image_base64' in result:
                    # Get user session data
                    user_data = user_sessions.get(session_id)
                    if user_data:
                        # Decode and save image
                        image_data = base64.b64decode(result['image_base64'])
                        image = Image.open(io.BytesIO(image_data))

                        # Get prompt and parent from session
                        prompt = user_data.get('pending_prompt', 'Generated image')
                        parent = user_data.get('pending_parent')

                        # Save image
                        image_name = save_image(session_id, image, prompt, parent)
                        increment_user_count(session_id)

                        # Clear pending data
                        user_data.pop('current_task_id', None)
                        user_data.pop('pending_prompt', None)
                        user_data.pop('pending_parent', None)

                        # Emit final result to frontend client
                        socketio.emit('image_ready', {
                            'image': image_name,
                            'prompt': prompt
                        }, room=session_id)

            @backend_sio.on('image_processing_error')
            def on_processing_error(data):
                """Relay error event to frontend"""
                task_id = data.get('task_id')
                error = data.get('error')
                session_id = find_session_for_task(task_id)
                if session_id:
                    socketio.emit('backend_error', {
                        'task_id': task_id,
                        'error': error
                    }, room=session_id)

            # Connect to backend
            backend_sio.connect(BACKEND_URL)

        except Exception as e:
            print(f"Error connecting to backend socket: {e}")
            backend_connected = False

def find_session_for_task(task_id: str) -> Optional[str]:
    """Find which user session owns a task"""
    for session_id, session_data in user_sessions.items():
        if session_data.get('current_task_id') == task_id:
            return session_id
    return None

def poll_backend_health():
    """Background task to poll backend health every 5 minutes"""
    import time
    while True:
        check_backend_availability()
        time.sleep(300)  # 5 minutes

# Check backend on startup
check_backend_availability()

# Start background health check polling
health_check_thread = Thread(target=poll_backend_health, daemon=True)
health_check_thread.start()

@app.route("/kuvagen/")
def index():
    """Serve main page"""
    return render_template("index.html")

@app.route("/kuvagen/api/check_backend")
def api_check_backend():
    """Check if local backend is available (returns cached status)"""
    with backend_status_lock:
        available = backend_available
    return jsonify({"available": available})

@app.route("/kuvagen/api/session_info", methods=["GET", "POST"])
def api_session_info():
    """Get current session information"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)

    with backend_status_lock:
        available = backend_available

    return jsonify({
        "user_id": user_id,
        "images_generated": user_data["images_generated"],
        "max_images": MAX_IMAGES_WITHOUT_LOGIN,
        "can_generate": check_user_limit(user_id),
        "backend_available": available,
        "default_model": user_data["selected_model"]
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected')
    # Join user to their room if they have a user_id
    user_id = get_user_id()
    if user_id:
        join_room(user_id)
        print(f'User {user_id} joined room')
    emit("status", "Connected to server! Ready to go!")

@socketio.on('user_id')
def handle_user_id(data):
    """Handle user ID from client"""
    if data:
        session['user_id'] = data
        user_data = get_or_create_session(data)
        # Join user to their own room for targeted emissions
        join_room(data)
        print(f'User ID set: {data}, images generated: {user_data["images_generated"]}')

@socketio.on("lang_select")
def handle_lang_select(data):
    """Handle language selection"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    user_data["transcription_language"] = None if data == "" else data
    print(f"User {user_id} selected language: {data}")
    emit("status", "Updated language selection!")

@socketio.on("mdl_select")
def handle_model_select(data):
    """Handle model selection"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    user_data["selected_model"] = data
    print(f"User {user_id} selected model: {data}")
    emit("status", "Updated model selection!")

@socketio.on("full_audio_data")
def handle_full_audio_data(data):
    """Handle audio transcription"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]
    language = user_data["transcription_language"]

    try:
        if "local" in selected_model and backend_available:
            # Use local backend
            response = requests.post(
                f"{BACKEND_URL}/transcribe",
                json={
                    "audio_base64": data,
                    "language": language
                },
                timeout=30
            )
            result = response.json()

            if result.get("error"):
                emit("empty_transcription", result["error"])
                emit("status", "Waiting...")
                return

            transcription = result.get("transcription", "")

        elif "cloud" in selected_model:
            # Use cloud transcription
            audio_bytes = base64.b64decode(data)

            # Save temporarily in temp_audio directory
            # Use .bin extension initially, OpenAI Whisper can handle various formats
            temp_path = os.path.join(TEMP_AUDIO_DIR, f"temp_audio_{uuid.uuid4()}.webm")
            try:
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)

                # Transcribe with OpenAI (it auto-detects format)
                with open(temp_path, "rb") as audio_file:
                    response = OPENAI_CLIENT.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language,
                        response_format="text"
                    )
                transcription = response
            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception as e:
                        print(f"Error removing temp audio: {e}")
        else:
            emit("error", "No transcription service available")
            return

        if transcription == "":
            emit("empty_transcription", "No audio detected, please try again.")
            emit("status", "Waiting...")
            return

        print(f"Transcription for {user_id}: {transcription}")
        emit("final_transcription_result", transcription)

    except Exception as e:
        print(f"Transcription error: {e}")
        emit("error", f"Transcription error: {str(e)}")

@socketio.on("audio_data")
def handle_audio_data(data):
    """Handle partial audio transcription for live feedback"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]
    language = user_data["transcription_language"]

    try:
        if "local" in selected_model and backend_available:
            # Use local backend
            response = requests.post(
                f"{BACKEND_URL}/transcribe",
                json={
                    "audio_base64": data,
                    "language": language
                },
                timeout=10
            )
            result = response.json()

            if not result.get("error"):
                transcription = result.get("transcription", "")
                if transcription and transcription.strip():
                    print(f"Partial transcription: {transcription}")
                    emit("transcription", transcription)

        elif "cloud" in selected_model:
            # Use cloud transcription
            audio_bytes = base64.b64decode(data)

            # Save temporarily in temp_audio directory
            temp_path = os.path.join(TEMP_AUDIO_DIR, f"temp_partial_{user_id}_{uuid.uuid4()}.wav")
            try:
                with open(temp_path, "wb") as f:
                    f.write(audio_bytes)

                with open(temp_path, "rb") as audio_file:
                    response = OPENAI_CLIENT.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text",
                        language=language if language else None
                    )
                    transcription = response if isinstance(response, str) else response.text

                    if transcription and transcription.strip():
                        print(f"Partial transcription: {transcription}")
                        emit("transcription", transcription)
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass

    except Exception as e:
        # Silently fail for partial transcriptions
        print(f"Partial transcription error (ignoring): {e}")
        pass

@socketio.on("process_command")
def handle_process_command(data):
    """Process user command through LLM and execute action"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)

    command = data.get("command")
    image = data.get("image")
    selected_model = user_data["selected_model"]

    print(f"Processing command for {user_id}: {command}")

    try:
        # Check user limit for cloud models only
        if "cloud" in selected_model and not check_user_limit(user_id):
            emit("command_error", "Image generation limit reached for cloud models. Please log in to continue or use local models.")
            return

        # Get LLM response for action determination
        if "local" in selected_model and backend_available:
            # Use local LLM
            response = requests.post(
                f"{BACKEND_URL}/llm",
                json={"user_prompt": command},
                timeout=30
            )
            llm_result = response.json()
            action = llm_result["action"]
            prompt = llm_result["prompt"]
        else:
            # Use cloud LLM
            action, prompt = poll_llm_cloud(command)

        print(f"LLM response: {action}, {prompt}")
        emit("llm_response", f"{action}: {prompt}")

        # Store prompt for later use
        user_data['pending_prompt'] = prompt

        # Execute action
        if action == "create":
            emit("status", "Creating new image...")
            user_data['pending_parent'] = None
            result = generate_image_socketio(user_id, prompt)
            if result:  # Cloud model returns immediately
                emit("command_result", result)
            # else: local model will emit via socket events
        elif action == "edit":
            emit("status", "Editing image...")
            user_data['pending_parent'] = image
            result = edit_image_socketio(user_id, image, prompt)
            if result:  # Cloud model returns immediately
                emit("command_result", result)
            # else: local model will emit via socket events
        elif action == "undo":
            emit("status", "Reverting to previous image...")
            result = previous_image_socketio(user_id, image)
            emit("command_result", result)
        elif action == "error":
            emit("command_error", prompt)
        else:
            emit("command_error", "Unknown action")

    except Exception as e:
        print(f"Error processing command: {e}")
        emit("command_error", str(e))

def poll_llm_cloud(user_prompt: str):
    """Use cloud LLM for action determination"""
    system_prompt = Path('intention_recognition_prompt_v3_no_video.txt').read_text()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = OPENAI_CLIENT.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=200,
            response_format=LLMOutput,
        )
        result = response.choices[0].message.parsed
        return result.action, result.prompt
    except Exception as e:
        print(f"Cloud LLM error: {e}")
        return "error", str(e)

def generate_image_socketio(user_id: str, prompt: str):
    """Generate new image - SocketIO version"""
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]

    if "local" in selected_model and backend_available:
        # Use local backend with queue
        response = requests.post(
            f"{BACKEND_URL}/generate_image",
            json={
                "prompt": prompt,
                "session_id": user_id
            },
            timeout=10
        )
        result = response.json()
        task_id = result["task_id"]

        # Track task for this session
        user_data['current_task_id'] = task_id

        # Join backend session room for receiving updates
        if backend_sio and backend_connected:
            backend_sio.emit('join_session', {'session_id': user_id})

        # Emit initial status to frontend
        socketio.emit('task_queued', {
            'task_id': task_id,
            'position': result.get('position', 0)
        }, room=user_id)

        # The actual result will come via socket events
        # We'll handle completion in the socket event handlers
        return None

    else:
        # Use cloud service
        image = None
        if CLOUD_PROVIDER == "openai":
            response = OPENAI_CLIENT.images.generate(
                prompt=prompt,
                model="dall-e-3",
                size="1024x1024",
                response_format="b64_json",
                quality="standard",
                n=1,
            )
            image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        elif CLOUD_PROVIDER == "google":
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please generate the following image: " + prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    break

        if image is None:
            raise Exception("No image generated")

        # Save image
        image_name = save_image(user_id, image, prompt, None)
        increment_user_count(user_id)

        return {"image": image_name, "prompt": prompt}

def edit_image_socketio(user_id: str, parent_image: str, prompt: str):
    """Edit existing image - SocketIO version"""
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]

    # Get parent image
    image = get_saved_image(user_id, parent_image)

    if "local" in selected_model and backend_available:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Determine model type
        model_type = "fast" if "fast" in selected_model else "slow"

        # Use local backend with queue
        response = requests.post(
            f"{BACKEND_URL}/edit_image",
            json={
                "prompt": prompt,
                "image_base64": image_base64,
                "session_id": user_id,
                "model": model_type
            },
            timeout=10
        )
        result = response.json()
        task_id = result["task_id"]

        # Track task for this session
        user_data['current_task_id'] = task_id
        user_data['pending_prompt'] = prompt
        user_data['pending_parent'] = parent_image

        # Join backend session room for receiving updates
        if backend_sio and backend_connected:
            backend_sio.emit('join_session', {'session_id': user_id})

        # Emit initial status to frontend
        socketio.emit('task_queued', {
            'task_id': task_id,
            'position': result.get('position', 0)
        }, room=user_id)

        # The actual result will come via socket events
        return None

    else:
        # Use cloud service
        edited_image = None
        if CLOUD_PROVIDER == "openai":
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, 'temp_image.png')
                image = image.convert('RGBA')
                image.save(temp_file_path, format='PNG')
                with open(temp_file_path, 'rb') as image_file, open("dalle2_mask.png", "rb") as mask_file:
                    response = OPENAI_CLIENT.images.edit(
                        prompt=prompt,
                        image=image_file,
                        mask=mask_file,
                        model="dall-e-2",
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                edited_image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        elif CLOUD_PROVIDER == "google":
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please edit the image: " + prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    edited_image = Image.open(io.BytesIO(part.inline_data.data))
                    break

        if edited_image is None:
            raise Exception("No edited image generated")

        # Save image
        image_name = save_image(user_id, edited_image, prompt, parent_image)
        increment_user_count(user_id)

        return {"image": image_name, "prompt": prompt}

def previous_image_socketio(user_id: str, image_name: str):
    """Go to previous image in history - SocketIO version"""
    user_data = get_or_create_session(user_id)
    gallery = user_data["gallery"]

    # Find current image in gallery
    for img_data in gallery:
        if img_data["name"] == image_name:
            parent = img_data.get("parent")
            if parent:
                return {"image": parent}
            else:
                return {"error": "No previous image"}

    return {"error": "No previous image"}

def poll_task_completion_socketio(user_id: str, task_id: str, prompt: str, parent: Optional[str]):
    """Poll backend task until completion - SocketIO version"""
    from flask_socketio import emit
    max_polls = 120  # 2 minutes max
    poll_count = 0

    while poll_count < max_polls:
        time.sleep(1)
        poll_count += 1

        try:
            response = requests.get(f"{BACKEND_URL}/task_status/{task_id}", timeout=5)
            status_data = response.json()

            status = status_data.get("status")

            if status == "processing":
                # Calculate progress percentage from step info
                progress_info = status_data.get("progress")
                if progress_info:
                    step = progress_info.get("step", 0)
                    total_steps = progress_info.get("total_steps", 1)
                    progress_percentage = int((step / total_steps) * 100)
                    emit("generation_progress", {"progress": progress_percentage})

                # Send preview if available
                preview = status_data.get("preview")
                if preview:
                    emit("generation_preview", {"preview": preview})
                    # Also emit as image_progress for compatibility
                    emit("image_progress", preview)

            elif status == "completed":
                result = status_data.get("result", {})
                image_base64 = result.get("image_base64")

                if not image_base64:
                    return {"error": "No image in result"}

                # Decode and save image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                image_name = save_image(user_id, image, prompt, parent)
                increment_user_count(user_id)

                return {"image": image_name, "prompt": prompt}

            elif status == "failed":
                error = status_data.get("error", "Unknown error")
                return {"error": error}

        except Exception as e:
            print(f"Error polling task: {e}")

    return {"error": "Task timeout"}

def generate_image(user_id: str, prompt: str):
    """Generate new image"""
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]

    if "local" in selected_model and backend_available:
        # Use local backend with queue
        response = requests.post(
            f"{BACKEND_URL}/generate_image",
            json={
                "prompt": prompt,
                "session_id": user_id
            },
            timeout=10
        )
        result = response.json()
        task_id = result["task_id"]
        position = result["position"]

        # Poll for completion
        return poll_task_completion(user_id, task_id, prompt, None)

    else:
        # Use cloud service
        image = None
        if CLOUD_PROVIDER == "openai":
            response = OPENAI_CLIENT.images.generate(
                prompt=prompt,
                model="dall-e-3",
                size="1024x1024",
                response_format="b64_json",
                quality="standard",
                n=1,
            )
            image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        elif CLOUD_PROVIDER == "google":
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please generate the following image: " + prompt],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(io.BytesIO(part.inline_data.data))
                    break

        if image is None:
            raise Exception("No image generated")

        # Save image
        image_name = save_image(user_id, image, prompt, None)
        increment_user_count(user_id)

        return jsonify({"image": image_name, "prompt": prompt})

def edit_image(user_id: str, parent_image: str, prompt: str):
    """Edit existing image"""
    user_data = get_or_create_session(user_id)
    selected_model = user_data["selected_model"]

    # Get parent image
    image = get_saved_image(user_id, parent_image)

    if "local" in selected_model and backend_available:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Determine model type
        model_type = "fast" if "fast" in selected_model else "slow"

        # Use local backend with queue
        response = requests.post(
            f"{BACKEND_URL}/edit_image",
            json={
                "prompt": prompt,
                "image_base64": image_base64,
                "session_id": user_id,
                "model": model_type
            },
            timeout=10
        )
        result = response.json()
        task_id = result["task_id"]

        # Poll for completion
        return poll_task_completion(user_id, task_id, prompt, parent_image)

    else:
        # Use cloud service
        edited_image = None
        if CLOUD_PROVIDER == "openai":
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, 'temp_image.png')
                image = image.convert('RGBA')
                image.save(temp_file_path, format='PNG')
                with open(temp_file_path, 'rb') as image_file, open("dalle2_mask.png", "rb") as mask_file:
                    response = OPENAI_CLIENT.images.edit(
                        prompt=prompt,
                        image=image_file,
                        mask=mask_file,
                        model="dall-e-2",
                        size="1024x1024",
                        response_format="b64_json",
                        n=1,
                    )
                edited_image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        elif CLOUD_PROVIDER == "google":
            response = GOOGLE_CLIENT.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=["Please edit the image: " + prompt, image],
                config=types.GenerateContentConfig(
                    response_modalities=['Text', 'Image']
                )
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    edited_image = Image.open(io.BytesIO(part.inline_data.data))
                    break

        if edited_image is None:
            raise Exception("No edited image generated")

        # Save image
        image_name = save_image(user_id, edited_image, prompt, parent_image)
        increment_user_count(user_id)

        return jsonify({"image": image_name, "prompt": prompt})

def poll_task_completion(user_id: str, task_id: str, prompt: str, parent: Optional[str]):
    """Poll backend task until completion"""
    max_polls = 120  # 2 minutes max
    poll_count = 0

    while poll_count < max_polls:
        try:
            response = requests.get(f"{BACKEND_URL}/task_status/{task_id}", timeout=5)
            status_data = response.json()

            status = status_data["status"]

            if status == "queued":
                position = status_data.get("position", 0)
                socketio.emit("status", f"In queue, position: {position + 1}")
            elif status == "processing":
                progress = status_data.get("progress")
                if progress:
                    step = progress["step"]
                    total = progress["total_steps"]
                    socketio.emit("status", f"Generating, Step {step}/{total}")

                # Send preview if available
                preview = status_data.get("preview")
                if preview:
                    socketio.emit("image_progress", preview)
            elif status == "completed":
                result = status_data["result"]
                image_base64 = result["image_base64"]

                # Decode and save image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))

                image_name = save_image(user_id, image, prompt, parent)
                increment_user_count(user_id)

                return jsonify({"image": image_name, "prompt": prompt})
            elif status == "error":
                error = status_data.get("error", "Unknown error")
                return jsonify({"error": error})

        except Exception as e:
            print(f"Error polling task: {e}")

        time.sleep(1)
        poll_count += 1

    return jsonify({"error": "Task timeout"})

def previous_image(user_id: str, image_name: str):
    """Go to previous image in history"""
    user_data = get_or_create_session(user_id)
    gallery = user_data["gallery"]

    # Find current image in gallery
    for img_data in gallery:
        if img_data["name"] == image_name:
            if img_data["parent"]:
                # Return parent
                for parent_data in gallery:
                    if parent_data["name"] == img_data["parent"]:
                        return jsonify({
                            "image": parent_data["name"],
                            "prompt": parent_data["prompt"],
                            "action": "undo"
                        })

    return jsonify({"error": "No previous image"})

def random_image_name(prompt: str, length: int = 6) -> str:
    """Generate random image name"""
    words = prompt.split()[:4]
    words = "-".join(words)
    random_name = ''.join(random.choices(string.ascii_letters, k=length))
    return words + "-" + random_name

def save_image(user_id: str, image: Image.Image, prompt: str, parent: Optional[str]) -> str:
    """Save image to user's gallery"""
    user_data = get_or_create_session(user_id)

    # Create directories
    gallery_dir = f"./gallery/{user_id}"
    os.makedirs(gallery_dir, exist_ok=True)
    os.makedirs(f"{gallery_dir}/thumbnails", exist_ok=True)

    # Generate name
    image_name = random_image_name(prompt)

    # Save full image
    image.save(f"{gallery_dir}/{image_name}.webp")

    # Save thumbnail
    thumb = image.resize((256, 256), Image.Resampling.LANCZOS)
    thumb.save(f"{gallery_dir}/thumbnails/{image_name}.webp")

    # Add to user's gallery
    user_data["gallery"].append({
        "name": image_name,
        "prompt": prompt,
        "parent": parent,
        "created_at": datetime.now().isoformat()
    })

    # Also save to global gallery.json for compatibility
    add_to_global_gallery(user_id, image_name, prompt, parent)

    return image_name

def add_to_global_gallery(user_id: str, name: str, prompt: str, parent: Optional[str]):
    """Add to global gallery.json (for backward compatibility)"""
    filename = "gallery.json"
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump([], file)

    with open(filename, 'r') as file:
        data = json.load(file)

    new_entry = {
        "name": name,
        "prompt": prompt,
        "parent": parent,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
    data.append(new_entry)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def get_saved_image(user_id: str, image_name: str) -> Image.Image:
    """Retrieve saved image"""
    # Try user's gallery first
    path = f"./gallery/{user_id}/{image_name}.webp"
    if os.path.exists(path):
        return Image.open(path)

    # Fall back to global gallery
    path = f"./gallery/{image_name}.webp"
    if os.path.exists(path):
        return Image.open(path)

    raise Exception(f"Image not found: {image_name}")

@app.route("/kuvagen/gallery")
def get_gallery_json():
    """Get user's gallery as JSON"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    return jsonify(user_data["gallery"])

@app.route("/kuvagen/images/<image>")
def get_image(image):
    """Serve image file"""
    user_id = get_user_id()

    # Try user's gallery
    user_path = f"./gallery/{user_id}"
    if os.path.exists(f"{user_path}/{image}.webp"):
        return send_from_directory(user_path, f"{image}.webp")

    # Fall back to global gallery
    return send_from_directory("./gallery", f"{image}.webp")

@app.route("/kuvagen/images")
def images():
    """Get user's images"""
    user_id = get_user_id()
    user_data = get_or_create_session(user_id)
    return jsonify(user_data["gallery"])

if __name__ == "__main__":
    print("Server started, ready to go!")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False)
