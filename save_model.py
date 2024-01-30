import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import ensure_path_exists

# Define a list of dictionaries for models
models = [
    {"friendly_name": "orca", "short_name": "Orca-2-7b", "vendor_name": "microsoft"},
    {"friendly_name": "tinyllama", "short_name": "TinyLlama-1.1B-Chat-v1.0", "vendor_name": "TinyLlama"},
    {"friendly_name": "phi2", "short_name": "phi-2", "vendor_name": "microsoft"},
]

# Set up argument parsing
parser = argparse.ArgumentParser(description='Download and save a specific model')
parser.add_argument('model_name', type=str, help='The friendly name of the model to download')
args = parser.parse_args()

# Find the model in the list
model_to_download = next((model for model in models if model['friendly_name'] == args.model_name), None)

if model_to_download:
    full_model_name = f"{model_to_download['vendor_name']}/{model_to_download['short_name']}"

    # Download and cache the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(full_model_name, trust_remote_code=True)
    model_instance = AutoModelForCausalLM.from_pretrained(full_model_name, trust_remote_code=True)

    # Create directory path
    model_directory = f"./models/{model_to_download['short_name'].lower()}"

    # Ensure the path exists
    ensure_path_exists(model_directory)

    # Save to the specified directory
    model_instance.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)
    print("Completed downloading model files.")
else:
    print(f"Model '{args.model_name}' not found.")