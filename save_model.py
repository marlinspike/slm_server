from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import ensure_path_exists

# Some models to try:
# phi-2
# Orca-2-7b
# Define the model name
model_short_name = "Orca-2-7b"
model_vendor_name = "microsoft"

model_short_name ="TinyLlama-1.1B-Chat-v1.0"
model_vendor_name = "TinyLlama"

model_short_name = "phi-2"
model_vendor_name = "microsoft"

model_name = f"{model_vendor_name}/{model_short_name}"


tokenizer = None
model = None

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

ensure_path_exists(f"./models/{model_short_name.lower()}")
# Save to the specified directory
model.save_pretrained(f"./models/{model_short_name.lower()}")
tokenizer.save_pretrained(f"./models/{model_short_name.lower()}")

