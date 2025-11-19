import base64
import json
import os
import time
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from pydantic import BaseModel

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

app = Flask(__name__, template_folder=".")
CORS(app)
socketio = SocketIO(app, async_mode="threading", path="/kuvagen/socket.io", cors_allowed_origins="*")

# --- Clients ---
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
GOOGLE_CLIENT = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# --- Session & Rate Limiting ---
class SessionManager:
    def __init__(self):
        self.sessions = {} # IP -> {count: int, last_active: float, last_image: str}
        self.limit = 10

    def check_limit(self, ip):
        if ip not in self.sessions:
            self.sessions[ip] = {"count": 0, "last_active": time.time(), "last_image": None}
        if self.sessions[ip]["count"] >= self.limit:
            return False
        return True

    def increment(self, ip):
        if ip not in self.sessions:
            self.sessions[ip] = {"count": 0, "last_active": time.time(), "last_image": None}
        self.sessions[ip]["count"] += 1
        self.sessions[ip]["last_active"] = time.time()
        return self.sessions[ip]["count"]

    def update_last_image(self, ip, image_path):
        if ip not in self.sessions:
            self.sessions[ip] = {"count": 0, "last_active": time.time(), "last_image": None}
        self.sessions[ip]["last_image"] = image_path

    def get_remaining(self, ip):
        if ip not in self.sessions:
            return self.limit
        return max(0, self.limit - self.sessions[ip]["count"])

    def get_last_image(self, ip):
        return self.sessions.get(ip, {}).get("last_image")

session_manager = SessionManager()

# --- Gallery Management ---
class GalleryManager:
    def __init__(self, gallery_file="gallery.json"):
        self.gallery_file = gallery_file
        self.ensure_gallery_file()

    def ensure_gallery_file(self):
        if not os.path.exists(self.gallery_file):
            with open(self.gallery_file, 'w') as f:
                json.dump([], f)

    def add_image(self, name, prompt, parent=None, ip=None):
        entry = {
            "name": name,
            "prompt": prompt,
            "parent": parent,
            "ip": ip,
            "timestamp": time.time()
        }
        with open(self.gallery_file, 'r+') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(entry)
            f.seek(0)
            json.dump(data, f, indent=4)
        return entry

    def get_gallery(self, ip=None):
        if not os.path.exists(self.gallery_file):
            return []
        with open(self.gallery_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return []
        if ip:
            return [img for img in data if img.get("ip") == ip]
        return data

gallery_manager = GalleryManager()

# --- LLM Models ---
class LLMOutput(BaseModel):
    action: str
    prompt: str

# --- Helper Functions ---
def check_backend_status():
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=2)
        return response.status_code == 200
    except:
        return False

def poll_llm_cloud(user_prompt, provider="openai"):
    system_prompt = "You are an AI assistant that identifies the user's intent from a voice command. Return a JSON object with 'action' (one of: create, edit, video, undo, error) and 'prompt' (the image generation prompt or error message)."
    try:
        if "openai" in provider and OPENAI_CLIENT:
            response = OPENAI_CLIENT.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=LLMOutput
            )
            result = response.choices[0].message.parsed
            return result.action, result.prompt
    except Exception as e:
        print(f"Error in Cloud LLM: {e}")
    return "error", str(e)

def save_and_emit_image(image_b64, prompt, ip, parent=None):
    # Generate filename without extension for ID
    image_id = f"img_{int(time.time())}_{ip.replace(':', '')}"
    filename = f"{image_id}.png"
    filepath = os.path.join("gallery", filename)
    os.makedirs("gallery", exist_ok=True)

    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_b64))

    # Update session
    session_manager.increment(ip)
    session_manager.update_last_image(ip, filename) # Store filename, not full path for easier retrieval

    # Update Gallery
    parent_id = None
    if parent:
        # Extract ID from filename if needed, but we store filenames as IDs mostly
        parent_id = os.path.splitext(os.path.basename(parent))[0]

    gallery_manager.add_image(image_id, prompt, parent=parent_id, ip=ip)

    # Emit with full filename for src
    emit("final_image", {"image": filename, "prompt": prompt, "remaining": session_manager.get_remaining(ip)})
    emit("status", "Ready")

def consume_backend_stream(response, ip, prompt, parent=None):
    """Consumes NDJSON stream from backend."""
    full_image_b64 = ""
    for line in response.iter_lines():
        if line:
            try:
                data = json.loads(line)
                status = data.get("status")
                if status == "queued":
                    emit("queue_status", {"position": data.get("position")})
                elif status == "progress":
                    # Backend might send partial images or just messages
                    # If it sends image progress:
                    if "image" in data:
                        emit("image_progress", data["image"])
                    if "message" in data:
                        emit("status", data["message"])
                elif status == "done":
                    full_image_b64 = data.get("image")
                    save_and_emit_image(full_image_b64, prompt, ip, parent)
                elif status == "error":
                    emit("error", data.get("message"))
            except json.JSONDecodeError:
                continue

def handle_create_image(prompt, mode, ip):
    if mode == "local":
        try:
            response = requests.post(f"{BACKEND_URL}/generate", json={"prompt": prompt}, stream=True)
            if response.status_code == 200:
                consume_backend_stream(response, ip, prompt)
            else:
                emit("error", f"Backend error: {response.text}")
        except Exception as e:
            emit("error", f"Connection error: {e}")
    else:
        # Cloud (OpenAI DALL-E 3)
        try:
            if not OPENAI_CLIENT:
                emit("error", "OpenAI API Key missing")
                return
            response = OPENAI_CLIENT.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json"
            )
            image_b64 = response.data[0].b64_json
            save_and_emit_image(image_b64, prompt, ip)
        except Exception as e:
            emit("error", f"Cloud generation error: {e}")

def handle_edit_image(last_image_filename, prompt, mode, ip):
    # We need the actual image data to send to backend/cloud
    # last_image_filename is just the filename in gallery/
    filepath = os.path.join("gallery", last_image_filename)
    if not os.path.exists(filepath):
        emit("error", "Original image file not found")
        return

    with open(filepath, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode('utf-8')

    if mode == "local":
        try:
            response = requests.post(f"{BACKEND_URL}/edit", json={"prompt": prompt, "image": image_b64}, stream=True)
            if response.status_code == 200:
                consume_backend_stream(response, ip, prompt, parent=last_image_filename)
            else:
                emit("error", f"Backend error: {response.text}")
        except Exception as e:
            emit("error", f"Connection error: {e}")
    else:
        # Cloud (OpenAI DALL-E 2 Edit - requires mask, simplified here to just generation or error)
        # DALL-E 2 edit requires a mask. For now, let's just say Cloud Edit is not fully supported or use variation?
        # Or use Gemini if available.
        emit("error", "Cloud editing not fully implemented (requires mask).")

# --- Socket Events ---

@socketio.on('connect')
def handle_connect():
    ip = request.remote_addr
    print(f'Client connected: {ip}')
    remaining = session_manager.get_remaining(ip)
    emit("session_info", {"remaining": remaining, "limit": session_manager.limit})
    backend_alive = check_backend_status()
    emit("backend_status", {"available": backend_alive})

@socketio.on("process_request")
def handle_process_request(payload):
    ip = request.remote_addr
    if not session_manager.check_limit(ip):
        emit("error", "Daily limit reached.")
        return

    mode = payload.get("mode", "cloud")
    audio_b64 = payload.get("audio")
    language = payload.get("language")

    print(f"Processing request from {ip} in {mode} mode")
    emit("status", "Transcribing...")

    # 1. Transcribe
    transcription = ""
    if mode == "local":
        try:
            audio_bytes = base64.b64decode(audio_b64)
            files = {'file': ('audio.webm', audio_bytes, 'audio/webm')}
            response = requests.post(f"{BACKEND_URL}/transcribe", files=files, data={"language": language})
            if response.status_code == 200:
                transcription = response.json().get("text", "")
            else:
                emit("error", f"Backend error: {response.text}")
                return
        except Exception as e:
            emit("error", f"Connection error: {e}")
            return
    else:
        try:
            if not OPENAI_CLIENT:
                emit("error", "OpenAI API Key missing")
                return
            temp_filename = f"temp_{ip}_{int(time.time())}.webm"
            with open(temp_filename, "wb") as f:
                f.write(base64.b64decode(audio_b64))
            with open(temp_filename, "rb") as audio_file:
                transcription = OPENAI_CLIENT.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language,
                    response_format="text"
                )
            os.remove(temp_filename)
        except Exception as e:
            emit("error", f"Cloud transcription error: {e}")
            return

    emit("transcription", transcription)
    emit("status", "Thinking...")

    # 2. LLM (Intention)
    action, prompt = "", ""
    if mode == "local":
        try:
            response = requests.post(f"{BACKEND_URL}/infer_action", json={"command": transcription})
            if response.status_code == 200:
                res = response.json()
                action, prompt = res.get("action"), res.get("prompt")
            else:
                emit("error", "Backend LLM error")
                return
        except Exception as e:
            emit("error", f"Backend LLM connection error: {e}")
            return
    else:
        action, prompt = poll_llm_cloud(transcription)

    emit("llm_response", f"{action}: {prompt}")

    if action == "error":
        emit("error", prompt)
        return

    # 3. Execute Action
    if action == "create":
        emit("status", "Generating image...")
        handle_create_image(prompt, mode, ip)
    elif action == "edit":
        emit("status", "Editing image...")
        last_image = session_manager.get_last_image(ip)
        if not last_image:
            emit("error", "No image to edit!")
            return
        handle_edit_image(last_image, prompt, mode, ip)
    elif action == "undo":
        emit("status", "Undo not implemented yet")

# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/kuvagen/images/<path:filename>")
def get_image(filename):
    if '.' not in filename:
        filename += '.png'
    return send_from_directory("gallery", filename)

@app.route("/kuvagen/images")
def get_images():
    ip = request.remote_addr
    images = gallery_manager.get_gallery(ip)
    return jsonify(images)

@app.route("/kuvagen/gallery")
def get_gallery_json():
    ip = request.remote_addr
    return jsonify(gallery_manager.get_gallery(ip))

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)