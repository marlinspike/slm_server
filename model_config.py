# Define a list of dictionaries for models
models = [
    {"friendly_name": "orca", "short_name": "Orca-2-7b", "vendor_name": "microsoft"},
    {"friendly_name": "tinyllama", "short_name": "TinyLlama-1.1B-Chat-v1.0", "vendor_name": "TinyLlama"},
    {"friendly_name": "phi2", "short_name": "phi-2", "vendor_name": "microsoft"},
]

def get_model_short_name(friendly_name):
    model_info = next((model for model in models if model['friendly_name'] == friendly_name), None)
    if model_info:
        return model_info['short_name']
    else:
        return None  # Handle case where friendly_name is not found