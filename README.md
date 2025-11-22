# Voice-Guided AI Imaging Platform

A web-based AI imaging platform that allows users to create and edit images using voice commands. The system supports both local and cloud-based processing with intelligent session management and queue systems.

## Architecture

The system has been completely refactored into a two-server architecture:

### 1. **Main Server (`main.py`)** - Lightweight Web Server & Relay

- Hosts the web interface (`index.html`)
- Manages user sessions and authentication
- Enforces image generation limits
- Routes requests to either local backend or cloud services
- Handles WebSocket connections for real-time updates
- **Runs on**: Any machine, minimal resources needed
- **Port**: 5001

### 2. **Backend Server (`backend.py`)** - Heavy Inference Processing

- Runs local AI models (Whisper, LLM, Image Generation/Editing)
- Implements queue system for image generation tasks
- Provides REST API endpoints
- Handles concurrent requests efficiently
- **Runs on**: Powerful machine with GPU
- **Port**: 8000

## Key Features

### Session Management

- **User Identification**: Users are identified by IP + User Agent hash
- **Image Limits**: Configurable limit (default: 10 images) before login required
- **Persistent Gallery**: User's images persist across sessions
- **Concurrent Users**: Multiple users can use the system simultaneously

### Processing Modes

#### Cloud Mode

- Uses OpenAI Whisper for transcription
- Uses OpenAI GPT-4 or Google Gemini for LLM
- Uses DALL-E 3 or Gemini for image generation/editing
- **No queuing** - all users get instant access to cloud services
- **No local backend required**

#### Local Mode (Fast)

- Uses Faster-Whisper for transcription (instant)
- Uses Ollama for LLM (instant with queue)
- Uses SDXL-Lightning for fast image generation (4 steps)
- Uses Instruct-Pix2Pix for fast image editing
- **Queue system** for image generation (one at a time)

#### Local Mode (Quality)

- Same transcription and LLM as Local Fast
- Uses SDXL-Lightning for image generation (8 steps)
- Uses COSXL-Edit for higher quality image editing
- **Queue system** with position tracking and progress updates

### Queue System

- Single image generation/editing task processed at a time
- Queue position displayed to users
- Real-time progress updates with partial image previews
- Task status: `queued` → `processing` → `completed`

### User Interface

- **Push-to-Talk Only**: Hold SPACE bar to record voice command
- **No VAD (Voice Activity Detection)**: Simplified, more reliable
- **No Video Generation**: Removed to streamline the system
- **Gallery View**: Browse all generated images with download buttons
- **Tree Chart View**: Visualize image evolution history
- **Session Info**: Display user's image count and limits

## Installation

### Main Server Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd voice-guided-imaging
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements-main.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the main server**

   ```bash
   python main.py
   ```

   The web interface will be available at `http://localhost:5001/kuvagen/`

### Backend Server Setup (Optional - for local inference)

1. **On your powerful GPU machine**

   ```bash
   cd voice-guided-imaging
   pip install -r requirements-backend.txt
   ```

2. **Configure environment**

   - Set `BACKEND_URL` in main server's `.env` to point to this machine
   - Set `HUGGINGFACE_TOKEN` for model downloads
   - Configure Ollama for local LLM

3. **Run the backend server**

   ```bash
   python backend.py
   ```

   The backend API will be available at `http://localhost:8000`

## Configuration

### Environment Variables

| Variable                   | Description                           | Default                        |
| -------------------------- | ------------------------------------- | ------------------------------ |
| `OPENAI_API_KEY`           | OpenAI API key for cloud services     | Required for cloud mode        |
| `GEMINI_API_KEY`           | Google Gemini API key                 | Required for Google cloud mode |
| `HUGGINGFACE_TOKEN`        | HuggingFace token for model downloads | Required for local mode        |
| `BACKEND_URL`              | URL of backend server                 | `http://localhost:8000`        |
| `CLOUD_PROVIDER`           | Cloud provider (`openai` or `google`) | `google`                       |
| `MAX_IMAGES_WITHOUT_LOGIN` | Image limit per user                  | `10`                           |
| `LOCAL_MODEL_SIZE`         | Whisper model size                    | `turbo`                        |
| `OLLAMA_MODEL`             | Ollama LLM model                      | See .env.example               |

### Deployment

#### Production Deployment

**Main Server (Web Hosting)**:

- Deploy to any web server (AWS, Google Cloud, Heroku, etc.)
- Minimal resources needed (1-2 GB RAM, 1 CPU)
- Use a proper WSGI server like Gunicorn:
  ```bash
  gunicorn -k geventwebsocket.gunicorn.workers.GeventWebSocketWorker -w 1 -b 0.0.0.0:5001 main:app
  ```

**Backend Server (GPU Machine)**:

- Deploy to GPU-enabled server (AWS EC2 with GPU, Google Cloud with GPU)
- Recommended: 16+ GB RAM, NVIDIA GPU with 8+ GB VRAM
- Run with Uvicorn:
  ```bash
  uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 1
  ```

## Usage

1. **Select Language**: Choose Auto, English, or Finnish
2. **Select Processing Mode**:
   - Local (Fast): Quick local processing
   - Local (Quality): Better quality, slower
   - Cloud: OpenAI/Google cloud services
3. **Hold SPACE bar** and speak your command
4. **Release SPACE** to process
5. **Commands**:
   - "Create a new image of a sunset"
   - "Edit the image to add mountains"
   - "Make it look like nighttime"
   - "Go back to the previous image"

## API Endpoints

### Main Server

- `GET /kuvagen/` - Web interface
- `GET /kuvagen/api/check_backend` - Check backend availability
- `GET /kuvagen/api/session_info` - Get user session info
- `POST /kuvagen/process_command` - Process voice command
- `GET /kuvagen/gallery` - Get user's gallery JSON
- `GET /kuvagen/images/<image>` - Serve image file
- `WebSocket /kuvagen/socket.io` - Real-time updates

### Backend Server

- `GET /health` - Health check
- `POST /transcribe` - Transcribe audio
- `POST /llm` - Process LLM request
- `POST /generate_image` - Queue image generation
- `POST /edit_image` - Queue image editing
- `GET /task_status/{task_id}` - Get task status

## Project Structure

```
voice-guided-imaging/
├── main.py                      # Lightweight web server & relay
├── backend.py                   # Heavy inference server
├── index.html                   # Web frontend (simplified)
├── requirements-main.txt        # Main server dependencies
├── requirements-backend.txt     # Backend server dependencies
├── .env.example                 # Environment configuration template
├── intention_recognition_prompt_v3_no_video.txt  # LLM system prompt
├── gallery/                     # User images (per-user folders)
│   ├── {user_id}/
│   │   ├── {image}.webp
│   │   └── thumbnails/
│   └── gallery.json            # Global gallery (backward compat)
└── cache/                       # Model cache directory
```

## Troubleshooting

### Backend Not Available

- Ensure backend server is running and accessible
- Check `BACKEND_URL` in main server's `.env`
- System automatically falls back to cloud mode if backend unavailable

### Image Limit Reached

- Users hit the `MAX_IMAGES_WITHOUT_LOGIN` limit
- Increase limit in `.env` or implement user authentication

### Slow Image Generation

- Check queue position in UI
- Only one image generated at a time in local mode
- Consider using cloud mode for multiple concurrent users

### Gallery Not Loading

- Check user session is active
- Verify gallery directory permissions
- Check browser console for errors

## Performance Considerations

### Cloud Mode

- ✅ No queuing or waiting
- ✅ Works for unlimited concurrent users
- ❌ Costs money per request
- ❌ Slower than local in some cases

### Local Mode

- ✅ Free after setup
- ✅ Fast transcription (Faster-Whisper)
- ✅ Fast LLM (Ollama with queue)
- ⚠️ Image generation queued (one at a time)
- ⚠️ Requires powerful GPU machine
- ❌ Queue builds up with many users

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please ensure:

1. Session handling works correctly for concurrent users
2. Queue system maintains order
3. No mixing of data between users
4. Gallery and tree view work with user-specific data

## Changelog

### Version 2.0 (Current)

- Complete architectural rework
- Separated main server and backend server
- Implemented session management with user IDs
- Added image generation limits
- Implemented queue system for local inference
- Removed VAD (Voice Activity Detection)
- Removed video generation
- Push-to-talk only mode
- Per-user gallery persistence
- Download buttons in gallery
- Concurrent user support
- Cloud services work simultaneously for all users
- Local services queued appropriately
