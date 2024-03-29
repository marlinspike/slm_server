from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import os
from dotenv import load_dotenv
import sys
from transformers import pipeline
from model_config import models, get_model_short_name
load_dotenv()

model_short_name = get_model_short_name("tinyllama") 
# Check if a command line argument is provided for the model name
if len(sys.argv) > 1:
    model_short_name = get_model_short_name(sys.argv[1])  # Use the first command line argument as the model name


MAX_TOKEN_RESPONSE = int(os.getenv("MAX_TOKEN_RESPONSE", "255"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_K = int(os.getenv("TOP_K","50"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
LOG_RESPONSES = os.getenv("LOG_RESPONSES", "False").lower() == "true"
print(f"Using Model: {model_short_name}")



logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the Phi-2 model from the local directory
if ( "phi-2" in model_short_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_short_name.lower()}/", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(f"./models/{model_short_name.lower()}/", trust_remote_code=True)
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}. \n Please ensure the model is downloaded and saved to the models directory.")
elif ( "TinyLlama" in model_short_name):
    try:
        tinyllama_pipe = pipeline("text-generation", model=f"./models/{model_short_name.lower()}/", device_map="auto")
    except Exception as e:
        logging.error(f"Error loading model: {e}. \n Please ensure the model is downloaded and saved to the models directory.")
        print(f"Error loading model: {e}")

def get_response_from_model(model_name: str, user_request: str, input_data):
    response = None
    logging.info(f"Running {model_name} with request: {request.json}")

    if(model_name == "phi-2"):
        input_data = request.json
        inputs = tokenizer(user_request, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=MAX_TOKEN_RESPONSE)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)        
    elif (model_name) == "TinyLlama-1.1B-Chat-v1.0":
        messages = [
            {"role": "system", "content": "You are a helpful chatbot",},
            {"role": "user", "content": user_request},
        ]
        prompt = tinyllama_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = tinyllama_pipe(prompt, max_new_tokens=MAX_TOKEN_RESPONSE, do_sample=True, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
        response = outputs[0]["generated_text"]

    if LOG_RESPONSES:    
        logging.info(f"Response from {model_name}: {response}")
    else:
        logging.info(f"Response from {model_name} not logged.")

    return jsonify(response)


    

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_data = request.json
    user_request = input_data['text']
    logging.info(f"Running predict() with request: {request.json}")
    return get_response_from_model(model_short_name, user_request, input_data)

@app.route('/test', methods=['POST', 'GET'])
def test():
    return jsonify({"response": "Hello World!"})

if __name__ == '__main__':
    port = os.getenv("PORT", "6002")
    print(f"Starting server on port {port}")
    app.run(debug=False, port=port)
