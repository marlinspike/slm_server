from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
model_short_name = "Orca-2-7b"
model_short_name ="TinyLlama-1.1B-Chat-v1.0"
model_short_name = "phi-2"
from transformers import pipeline

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the Phi-2 model
#tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)

# Load the Phi-2 model from the local directory
tokenizer = AutoTokenizer.from_pretrained(f"./models/{model_short_name.lower()}/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(f"./models/{model_short_name.lower()}/", trust_remote_code=True)

def get_response_from_model(model_name: str, user_request: str, input_data):
    if(model_name == "phi-2"):
        input_data = request.json
        inputs = tokenizer(user_request, return_tensors='pt')
        outputs = model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Response: {response}")
        return jsonify(response)
    elif (model_name) == "TinyLlama-1.1B-Chat-v1.0":
        pipe = pipeline("text-generation", model="TinyLlama-1.1B-Chat-v1.0", device_map="auto")
        # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        messages = [
            {
                "role": "system",
                "content": "You are a friendly chatbot",
            },
            {"role": "user", "content": user_request},
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return(outputs[0]["generated_text"])

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_data = request.json
    user_request = input_data['text']
    logging.info(f"Running predict() with request: {request.json}")
    return get_response_from_model(model_short_name, user_request, input_data)

    """
    input_data = request.json
    inputs = tokenizer(input_data['text'], return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logging.info(f"Response: {response}")
    return jsonify(response)
    """

@app.route('/test', methods=['POST'])
def test():
    return jsonify("Hello World")

if __name__ == '__main__':
    app.run(debug=False, port="6001")
