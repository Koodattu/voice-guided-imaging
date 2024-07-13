import sounddevice as sd
import numpy as np
import whisper
import torch
from scipy.io.wavfile import write
from pynput import keyboard

# Initialize Whisper model
whisper_model = whisper.load_model("base")

# Function to record audio
def record_audio(duration, samplerate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished")
    return audio

# Function to save audio to a WAV file
def save_audio_to_wav(audio, filename, samplerate=44100):
    write(filename, samplerate, audio)

# Function to transcribe and translate audio using Whisper
def transcribe_and_translate(model, audio_file):
    result = model.transcribe(audio_file, task="translate")
    return result['text']

# Function to process text with GPT-2
def process_text(transcribed_text):
    # TODO
    command = transcribed_text
    return command

# Main function with push-to-talk functionality
def main():
    duration = 5  # Record for 5 seconds
    recording = False
    audio_data = []

    def on_press(key):
        nonlocal recording, audio_data
        try:
            if key.char == 'r':
                if not recording:
                    print("Recording started...")
                    recording = True
                    audio_data = record_audio(duration)
                else:
                    print("Recording already in progress...")
        except AttributeError:
            pass

    def on_release(key):
        nonlocal recording
        if key.char == 'r' and recording:
            print("Recording stopped.")
            recording = False
            audio_file = 'recorded_audio.wav'
            save_audio_to_wav(audio_data, audio_file)
            text = transcribe_and_translate(whisper_model, audio_file)
            print("Transcribed and translated text:", text)
            command = process_with_gpt2(text)
            print("Generated command:", command)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    listener.join()

if __name__ == "__main__":
    main()
