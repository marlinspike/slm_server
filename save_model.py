from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model name
model_name = "microsoft/phi-2"

# Download and cache the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Save to the specified directory
model.save_pretrained('./models/phi-2')
tokenizer.save_pretrained('./models/phi-2')
