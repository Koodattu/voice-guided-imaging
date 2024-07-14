import sounddevice as sd
import numpy as np
import whisper
import torch
from scipy.io.wavfile import write
from pynput import keyboard
import threading
from queue import Queue
import time
import tkinter as tk
from PIL import Image, ImageTk

class WhisperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcription App")
        self.root.geometry("800x600")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.whisper_model = whisper.load_model("medium").to(self.device)

        self.recording = False
        self.audio_queue = Queue()
        self.buffer = []
        self.stop_event = threading.Event()

        self.setup_ui()
        self.setup_keyboard_listener()

    def setup_ui(self):
        self.status_label = tk.Label(self.root, text="Press 'R' to start/stop recording", font=("Helvetica", 16))
        self.status_label.pack(pady=20)

        self.transcription_label = tk.Label(self.root, text="", font=("Helvetica", 14), wraplength=700, justify="left")
        self.transcription_label.pack(pady=20)

        self.command_label = tk.Label(self.root, text="Command: ", font=("Helvetica", 16))
        self.command_label.pack(pady=20)

    def setup_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()

    def on_press(self, key):
        try:
            if key.char == 'r' and not self.recording:
                self.update_status("Recording...")
                print("Recording started...")
                self.recording = True
                self.transcription_thread = threading.Thread(target=self.handle_audio_stream)
                self.transcription_thread.start()
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            if key.char == 'r' and self.recording:
                self.update_status("Stopped recording")
                print("Recording stopped.")
                self.recording = False
                if self.transcription_thread is not None:
                    self.transcription_thread.join()
                if self.buffer:
                    audio_data = np.concatenate(self.buffer, axis=0)
                    audio_file = 'final_audio.wav'
                    self.save_audio_to_wav(audio_data, audio_file)
                    transcribed_text = self.transcribe_and_translate(audio_file)
                    processed_command = self.process_text(transcribed_text)
                    print(f"Processed Command: {processed_command}")
                    self.command_label.config(text=f"Command: {processed_command}")
        except AttributeError:
            pass

    def transcribe_and_translate(self, audio_file):
        print("Transcribing and translating audio...")
        transcription = self.whisper_model.transcribe(audio_file)
        language = transcription['language']
        print(f"Detected Language: {language}")
        print(f"Transcription: {transcription['text']}")

        if language == "en":
            return transcription['text']
        
        translation = self.whisper_model.transcribe(audio_file, task="translate")
        print(f"Translation: {translation['text']}")
        return translation['text']

    def process_text(self, transcribed_text):
        command = transcribed_text
        return command

    def transcribe_audio_stream(self):
        last_transcription_time = time.time()
        while not self.stop_event.is_set() or not self.audio_queue.empty():
            while not self.audio_queue.empty():
                chunk = self.audio_queue.get()
                self.buffer.append(chunk)

            current_time = time.time()
            if current_time - last_transcription_time >= 1:
                if self.buffer:
                    audio_data = np.concatenate(self.buffer, axis=0)
                    audio_file = 'temp_audio.wav'
                    self.save_audio_to_wav(audio_data, audio_file)
                    transcription = self.whisper_model.transcribe(audio_file)
                    self.update_live_transcription(transcription['text'])
                    print(f"Real-Time Transcription: {transcription['text']}")
                    last_transcription_time = current_time

    def save_audio_to_wav(self, audio, filename, samplerate=44100):
        write(filename, samplerate, audio)

    def record_audio_stream(self):
        print("Recording...")

        def callback(indata, frames, time, status):
            if self.recording:
                self.audio_queue.put(indata.copy())

        with sd.InputStream(callback=callback, samplerate=44100, channels=1, dtype='int16'):
            while self.recording:
                sd.sleep(100)

        print("Recording finished")

    def handle_audio_stream(self):
        self.buffer = []
        self.stop_event.clear()

        transcription_thread = threading.Thread(target=self.transcribe_audio_stream)
        transcription_thread.start()
        self.record_audio_stream()
        self.stop_event.set()
        transcription_thread.join()
        print("Transcription thread finished")

    def update_live_transcription(self, text):
        self.transcription_label.config(text=text)

    def update_status(self, text):
        self.status_label.config(text=text)

if __name__ == "__main__":
    print("Starting application!")
    root = tk.Tk()
    app = WhisperApp(root)
    root.mainloop()
