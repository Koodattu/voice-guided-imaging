### Step-by-Step Installation Guide
1. Set Up Python Environment
First, ensure you have Python 3.8 or later installed. You can download it from the official Python website.
```
python -m venv image_edit_venv
./image_edit_venv/Scripts/activate
```
2. Install Dependencies
Install necessary libraries including PyTorch, Whisper, and sound handling libraries:
```
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/whisper.git
pip install sounddevice numpy scipy pynput
```
### Running the Script
Activate your virtual environment:
```
source image_edit_env/bin/activate
```
Run the script:
```
python script_name.py
```
This setup ensures you have a robust system to handle audio recording, transcription, translation, and processing using LLMs.

### Explanation of the Script
1. Recording Audio:
record_audio: Records audio for a specified duration.
save_audio_to_wav: Saves the recorded audio to a WAV file.

2. Transcription and Translation with Whisper:
transcribe_and_translate: Uses Whisper to transcribe and translate the audio file.

3. Processing with LLM:
process_text: Takes the transcribed text and uses a LLM to generate a command based on the input text.

4. Push-to-Talk Functionality:
Uses the pynput library to detect when the 'r' key is pressed and released to start and stop recording.

5. Main Function:
Continuously listens for the 'r' key press to record audio, then transcribes, translates, and processes the text to generate commands.
