import sounddevice as sd
import numpy as np
import whisper
import torch
from scipy.io.wavfile import write
from pynput import keyboard
import threading
from queue import Queue
import time

print("Starting application!")

print("Checking for cuda")
# Check if a GPU is available and use it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Whisper model...")
# Initialize Whisper model and move it to the device
whisper_model = whisper.load_model("medium").to(device)

# Global variables to control recording
recording = False
audio_queue = Queue()
buffer = []
stop_event = threading.Event()

# Function to transcribe and translate audio using Whisper
def transcribe_and_translate(model, audio_file):
    print("Transcribing and translating audio...")
    transcription = model.transcribe(audio_file)
    language = transcription['language']
    print(f"Detected Language: {language}")
    print(f"Transcription: {transcription['text']}")

    if language == "en":
        return transcription['text']
    
    translation = model.transcribe(audio_file, task="translate")
    print(f"Translation: {translation['text']}")
    return translation['text']

# Function to process text with LLM
def process_text(transcribed_text):
    # Placeholder for processing text with another LLM
    command = transcribed_text
    return command

# Function to transcribe audio chunks in real-time at regular intervals
def transcribe_audio_stream():
    last_transcription_time = time.time()
    while not stop_event.is_set() or not audio_queue.empty():
        while not audio_queue.empty():
            chunk = audio_queue.get()
            buffer.append(chunk)

        # Transcribe every second
        current_time = time.time()
        if current_time - last_transcription_time >= 1:
            if buffer:
                audio_data = np.concatenate(buffer, axis=0)
                audio_file = 'temp_audio.wav'
                save_audio_to_wav(audio_data, audio_file)
                transcription = whisper_model.transcribe(audio_file)
                print(f"Real-Time Transcription: {transcription['text']}")
                last_transcription_time = current_time

# Function to save audio to a WAV file
def save_audio_to_wav(audio, filename, samplerate=44100):
    write(filename, samplerate, audio)

# Function to stream and record audio
def record_audio_stream():
    global recording
    print("Recording...")

    def callback(indata, frames, time, status):
        if recording:
            audio_queue.put(indata.copy())
    
    with sd.InputStream(callback=callback, samplerate=44100, channels=1, dtype='int16'):
        while recording:
            sd.sleep(100)

    print("Recording finished")

# Function to handle recording and transcription
def handle_audio_stream():
    global buffer, stop_event
    buffer = []  # Clear the buffer at the start of each recording session
    stop_event.clear()  # Reset the stop event

    transcription_thread = threading.Thread(target=transcribe_audio_stream)
    transcription_thread.start()
    record_audio_stream()
    stop_event.set()  # Signal the transcription thread to stop
    transcription_thread.join()
    print("Transcription thread finished")

# Main function with push-to-talk functionality
def main():
    global recording, transcription_thread, buffer, stop_event

    print("Current recording device:", sd.query_devices(sd.default.device['input'])['name'])
    print("Press 'R' to start/stop recording")
    
    def on_press(key):
        global recording, transcription_thread
        try:
            if key.char == 'r' and not recording:
                print("Recording started...")
                recording = True
                transcription_thread = threading.Thread(target=handle_audio_stream)
                transcription_thread.start()
        except AttributeError:
            pass

    def on_release(key):
        global recording, transcription_thread
        try:
            if key.char == 'r' and recording:
                print("Recording stopped.")
                recording = False
                if transcription_thread is not None:
                    transcription_thread.join()
                # Process the final transcription and translation
                if buffer:
                    audio_data = np.concatenate(buffer, axis=0)
                    audio_file = 'final_audio.wav'
                    save_audio_to_wav(audio_data, audio_file)
                    transcribed_text = transcribe_and_translate(whisper_model, audio_file)
                    processed_command = process_text(transcribed_text)
                    print(f"Processed Command: {processed_command}")
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

if __name__ == "__main__":
    main()
