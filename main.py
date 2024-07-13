import sounddevice as sd
import numpy as np
import whisper
import torch
from scipy.io.wavfile import write
from pynput import keyboard
import threading
from queue import Queue

# Check if a GPU is available and use it
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize Whisper model and move it to the device
whisper_model = whisper.load_model("small").to(device)

# Global variables to control recording
recording = False
audio_queue = Queue()

# Function to record audio
def record_audio(samplerate=44100):
    global recording, audio_data
    print("Recording...")
    audio = []

    def callback(indata, frames, time, status):
        if recording:
            audio.append(indata.copy())

    with sd.InputStream(callback=callback, samplerate=samplerate, channels=1, dtype='int16'):
        while recording:
            sd.sleep(100)
    print("Recording finished")
    audio_data = np.concatenate(audio, axis=0)
    audio_queue.put(audio_data)

# Function to save audio to a WAV file
def save_audio_to_wav(audio, filename, samplerate=44100):
    write(filename, samplerate, audio)

# Function to transcribe and translate audio using Whisper
def transcribe_and_translate(model, audio_file):
    print("Transcribing and translating audio...")
    transcription = model.transcribe(audio_file)
    language = transcription['language']
    print(f"Detected Language: {language}")
    print(f"Transcription: {transcription['text']}")

    if language == "en":
        return transcription['text']
    
    translation = model.transcribe(audio_file, language="en", task="translate")
    return translation['text']

# Function to process text with LLM
def process_text(transcribed_text):
    # TODO
    command = transcribed_text
    return command

# Main function with push-to-talk functionality
def main():
    global recording
    print("Current recording device:", sd.query_devices(sd.default.device['input']))

    def on_press(key):
        global recording
        try:
            if key.char == 'r' and not recording:
                print("Recording started...")
                recording = True
                threading.Thread(target=record_audio).start()
        except AttributeError:
            pass

    def on_release(key):
        global recording, audio_data
        try:
            if key.char == 'r' and recording:
                print("Recording stopped.")
                recording = False
                # Ensure the recording thread has finished
                threading.Thread(target=process_recording).start()
        except AttributeError:
            pass

    def process_recording():
        audio_data = audio_queue.get()
        audio_file = 'recorded_audio.wav'
        save_audio_to_wav(audio_data, audio_file)
        text = transcribe_and_translate(whisper_model, audio_file)
        print("Transcribed and translated text:", text)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

if __name__ == "__main__":
    main()
