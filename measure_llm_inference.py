import os
import time
import psutil
import pynvml
from langchain_community.chat_models import ChatOllama
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1) OPTIONAL: Load environment variables from a .env file
# ------------------------------------------------------------------
# If you're using a .env file to store sensitive information like tokens,
# uncomment the following line and ensure your .env file is correctly set up.
load_dotenv()

# ------------------------------------------------------------------
# 3) Initialize NVML (for GPU VRAM measurement)
# ------------------------------------------------------------------
pynvml.nvmlInit()
gpu_handle = None
try:
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Use GPU 0
except pynvml.NVMLError as e:
    print(f"Error accessing GPU 0: {e}")
    print("VRAM measurements will not be available.")

# ------------------------------------------------------------------
# 4) Memory Measurement Utilities
# ------------------------------------------------------------------
def get_ram_usage_gb():
    """Returns used RAM and total RAM in GB."""
    vm = psutil.virtual_memory()
    used_gb = vm.used / (1024**3)
    total_gb = vm.total / (1024**3)
    return used_gb, total_gb

def get_vram_usage_gb(handle):
    """Returns used VRAM and total VRAM in GB for given GPU handle."""
    if not handle:
        return None, None
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = mem_info.used / (1024**3)
        total_gb = mem_info.total / (1024**3)
        return used_gb, total_gb
    except pynvml.NVMLError as error:
        print(f"Error retrieving VRAM usage: {error}")
        return None, None

def print_usage(stage):
    """Prints RAM and VRAM usage at a given stage."""
    used_ram, total_ram = get_ram_usage_gb()
    used_vram, total_vram = get_vram_usage_gb(gpu_handle)

    if used_vram is not None:
        print(
            f"{stage} - "
            f"RAM: {used_ram:.2f}/{total_ram:.2f} GB | "
            f"VRAM: {used_vram:.2f}/{total_vram:.2f} GB"
        )
    else:
        print(
            f"{stage} - "
            f"RAM: {used_ram:.2f}/{total_ram:.2f} GB | "
            f"VRAM: N/A"
        )

# ------------------------------------------------------------------
# 5) LLM Initialization
# ------------------------------------------------------------------
print("\nLoading LLM...")
llm = ChatOllama(model="mistral:instruct")
ready_response = llm.invoke("Respond with: Mistral-instruct ready to server!").content
print(f"LLM Response: {ready_response}\n")

# ------------------------------------------------------------------
# 6) Define Instruction Prompt and User Inputs
# ------------------------------------------------------------------
instruction_prompt = """
You are an assistant for image creation and editing tasks.
Users will provide inputs to create, edit, or undo images.
Correct any typos or unclear language to make the prompt suitable for stable diffusion models.
The user input might say something like "in the image" but don't include that in the prompt.
Always respond in a single JSON object with the appropriate field based on the task.
If the input is empty or short or is unrelated to image creation or editing, respond with "unknown".
The json "action" key's value must always be either "create", "edit", "video", "undo" or "unknown".
Don't return "create" unless the user input explictly asks for a new image.
Don't return "video" unless the user input explictly asks for a video to be generated of the current image. 
Make sure that you always respond in valid json format.

Examples:
User Input: "create me a new picture which has a horse on road"
Response: { "action":"create", "prompt":"a horse on a road" }

User Input: "make new image man driving motorcycle"
Response: { "action":"create", "prompt":"a man driving a motorcycle" }

User Input: "edit the image so that the horse is a cow"
Response: { "action":"edit", "prompt":"change the horse to a cow" }

User Input: "make it rain"
Response: { "action":"edit", "prompt":"make it rain" }

User Input: "add a car to the current image"
Response: { "action":"edit", "prompt":"add car" }

User Input: "remove chair from the image"
Response: { "action":"edit", "prompt":"remove the chair" }

User Input: "edit the image so the mountains in the back are replaced with a city skyline"
Response: { "action":"edit", "prompt":"replace the mountains with a city skyline" }

User Input: "add boats to the lake in the image"
Response: { "action":"edit", "prompt":"add boats on the water" }

User Input: "it is now night"
Response: { "action":"edit", "prompt":"it is now night" }

User Input: "time of day night"
Response: { "action":"edit", "prompt":"make night time" }

User Input: "add fireworks to the sky in the image"
Response: { "action":"edit", "prompt":"add fireworks to the sky" }

User Input: "edit the image and make them look like flight attendants"
Response: { "action":"edit", "prompt":"make them look like flight attendants" }

User Input: "make them look like doctors"
Response: { "action":"edit", "prompt":"make them look like doctors" }

User Input: "make her wear sunglasses"
Response: { "action":"edit", "prompt":"put on a pair of sunglasses" }

User Input: "add a train into the image"
Response: { "action":"edit", "prompt":"insert a train" }

User Input: "add a bridge over the river"
Response: { "action":"edit", "prompt":"insert a bridge over the river" }

User Input: "move them to space"
Response: { "action":"edit", "prompt":"put them in outer space" }

User Input: "make it look like its realistic"
Response: { "action":"edit", "prompt":"convert to a realistic photo" }

User Input: "make it 3d"
Response: { "action":"edit", "prompt":"turn into a 3d model" }

User Input: "in a race car video game"
Response: { "action":"edit", "prompt":"in a race car video game" }

User Input: "swap the sunflowers in the image with roses"
Response: { "action":"edit", "prompt":"swap sunflowers with roses" }

User Input: "what would it look like if it was a western movie"
Response: { "action":"edit", "prompt":"turn it into a still from a western" }

User Input: "make the image look like an oil pastel drawing"
Response: { "action":"edit", "prompt":"turn into an oil pastel drawing" }

User Input: "change the image so that the man is a woman"
Response: { "action":"edit", "prompt":"change the man into a woman" }

User Input: "make video from the image"
Response: { "action":"video", "prompt":"Generating a video" }

User Input: "create a video from this"
Response: { "action":"video", "prompt":"Generating a video" }

User Input: "let's go to the previous image"
Response: { "action":"undo", "prompt":"Reverting last action" }

User Input: "let's go back"
Response: { "action":"undo", "prompt":"Reverting last action" }

User Input: "What's the weather like today?"
Response: { "action":"unknown", "prompt":"Command was not recognized as suitable" }

User Input: "whats going on whats up"
Response: { "action":"unknown", "prompt":"Command was not recognized as suitable" }

Now handle the following:

User Input: "<user_input>"
Response:
"""

user_inputs = [
    "create me a new picture which has a horse on road",
    "make new image man driving motorcycle",
    "edit the image so that the horse is a cow",
    "make it rain",
    "add a car to the current image",
    "remove chair from the image",
    "edit the image so the mountains in the back are replaced with a city skyline",
    "add boats to the lake in the image",
    "it is now night",
    "time of day night",
    "add fireworks to the sky in the image",
    "edit the image and make them look like flight attendants",
    "make them look like doctors",
    "make her wear sunglasses",
    "add a train into the image",
    "add a bridge over the river",
    "move them to space",
    "make it look like its realistic",
    "make it 3d",
    "in a race car video game",
    "swap the sunflowers in the image with roses",
    "what would it look like if it was a western movie",
    "make the image look like an oil pastel drawing",
    "change the image so that the man is a woman",
    "make video from the image",
    "create a video from this",
    "let's go to the previous image",
    "let's go back",
    "What's the weather like today?",
    "whats going on whats up"
]

# ------------------------------------------------------------------
# 7) Benchmark Function
# ------------------------------------------------------------------
def benchmark_llm(llm, inputs, instruction):
    """
    Sends a list of inputs to the LLM with a predefined instruction prompt,
    measures response times, and calculates the average inference speed.
    """
    total_time = 0.0
    response_times = []
    responses = []

    print("Starting LLM inference benchmark...\n")

    for idx, user_input in enumerate(inputs, 1):
        print(f"Test {idx}/{len(inputs)}:")
        print(f"User Input: \"{user_input}\"")

        # Prepare the full prompt by combining instruction and user input
        full_prompt = f"{instruction}\n\nUser Input: \"{user_input}\"\nResponse:"

        # Measure time
        start_time = time.time()
        response = llm.invoke(full_prompt).content.strip()
        end_time = time.time()

        elapsed = end_time - start_time
        response_times.append(elapsed)
        total_time += elapsed
        responses.append(response)

        print(f"Response: {response}")
        print(f"Time Taken: {elapsed:.4f} seconds\n")

    average_time = total_time / len(inputs) if inputs else 0
    print(f"Benchmark Completed.\nAverage Inference Time: {average_time:.4f} seconds over {len(inputs)} tests.")

    return responses, response_times, average_time

# ------------------------------------------------------------------
# 8) Execute Benchmark
# ------------------------------------------------------------------
def main():
    # Initial Memory Usage
    print_usage("Initial State")

    # Define the instruction prompt
    instruction = instruction_prompt.strip()

    # Run Benchmark
    responses, times, avg_time = benchmark_llm(llm, user_inputs, instruction)

    # Final Memory Usage
    print_usage("Final State")

    # Optionally, save responses and times to a file or process further
    # For simplicity, we'll skip that step here.

if __name__ == "__main__":
    main()

    # Shutdown NVML
    pynvml.nvmlShutdown()
