from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the Phi-2 model
#tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
#model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True)
# Load the Phi-2 model from the local directory
tokenizer = AutoTokenizer.from_pretrained("./models/phi-2/", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./models/phi-2/", trust_remote_code=True)

@app.route('/predict', methods=['POST'])
def predict():
    logging.info(f"Running predict() with request: {request.json}")
    input_data = request.json
    inputs = tokenizer(input_data['text'], return_tensors='pt')
    outputs = model.generate(inputs['input_ids'], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify(response)

@app.route('/test', methods=['POST'])
def test():
    return jsonify("Hello World")

if __name__ == '__main__':
    app.run(debug=True, port="6001")
