from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path: str, model_name: str, device: str, prompt: str, data: dict):
    
    model_namepath = model_path + model_name

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_namepath)
    model = AutoModelForCausalLM.from_pretrained(model_namepath, device_map="auto", torch_dtype="auto").to(device)

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if data is None:
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    else:
        generated_ids = model.generate(**inputs, **data, max_new_tokens=100)


