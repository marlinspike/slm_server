## Phi-2 Language Server
This project lets you use the Microsoft Phi-2 (or any other Huggingface model), from a local repository. The project includes an app to download the model and a language server to use it.

### Setup
1. Clone the repository
2. Install the requirements
3. Create a .env file, using the .env.example file as a template
4. Run `save_model.py` to download the model
```
Some models to try:
- Orca
model_short_name = "Orca-2-7b"
model_vendor_name = "microsoft"

- TinyLlama
model_short_name ="TinyLlama-1.1B-Chat-v1.0"
model_vendor_name = "TinyLlama"

- Phi-2
model_short_name = "phi-2"
model_vendor_name = "microsoft"
```
4. Run `server.py` to start the language server


### Usage
Once the language server is running, use the endpoint exposed (by default on port 6001), to hit the 'predict' endpoint.
