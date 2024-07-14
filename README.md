### UCS LLM Voice Image Edit
This repository contains the code and guide for LLM application for generating and editing images via voice commands, showcased at the Tietoprovinssi event in Frami X, Seinäjoki.

### Step-by-Step Guide
1. Open terminal in the directory where you cloned the repository
2. Set up and activate Python virtual environment
```
python -m venv image_edit_venv
./image_edit_venv/Scripts/activate
```
2. Install dependencies with pip
```
pip install -r requirements.txt
```
3. Run the script:
```
python main.py
```
4. Hold the key ```R``` on the keyboard to start recording audio and release it to stop and process the given command.

### Application Logic Explained
1. The default microphone is used to record voice audio with push-to-talk functionality.
2. The audio is transcribed and translated to english (if necessary)
3. A LLM is used to turn the natural language input into a command for either Lightning-SDXL or ImaginAlry
4. The LLM also decides if the input suggest that a new image should be generated, if the current image should be edited or if it is unable to process the given input
