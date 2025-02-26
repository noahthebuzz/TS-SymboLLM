import json
import os

def write_json(data: dict, path: str, overwrite: bool) -> bool:
    try:
        if not overwrite:
            if os.path.exists(path):
                old_data = read_json(path=path)
                if old_data:
                    for key in old_data.keys():
                        if key not in data.keys():
                            data[key] = old_data[key]
        
        with open(path, 'w') as file:
            json.dump(data, file, indent=4)
        return True
    except Exception as e:
        return False


def read_json(path: str) -> dict | None:
    try:
        if os.path.exists(path):
            with open(path, 'r') as file:
                return json.load(file)
        return None
    except Exception as e:
        print(f"[ERROR]:\n{e}")
        return None
    

def get_config_content(setup_usr: str = None, logs: bool = None, ollama_param: str = None, models: list[str] = None) -> dict | None:
    if setup_usr is None and logs is None and ollama_param is None and models is None:
        return None
    try:
        data = read_json("src/config/config.json")
        new_data = {}

        # USER
        if setup_usr:
            new_data.update({setup_usr: data.get("setup").get(setup_usr)})

        # LOGS
        if logs:
            if data.get("logs") is not None:
                new_data.update({"logs": data.get("logs")})
            else:
                new_data.update({"logs": 0})

        # OLLAMA PARAMETER
        if ollama_param:
            if ollama_param == "rational":
                new_data.update({"params": data.get("ollama_parameter").get("rational")})
            elif ollama_param == "creative":
                new_data.update({"params": data.get("ollama_parameter").get("creative")}) 

        # MODELS
        if models is not None:
            model_list = []
            for model in models:
                for x in data.get("models").get(model):
                    model_list.append(x)
                new_data.update({"models": model_list})

        return new_data
    except Exception as e:
        print(f"[ERROR]:\n{e}")
        return None
    

def get_ollama_parameter(isRational: bool) -> dict:
    options = get_config_content(ollama_param="rational" if isRational else "creative")
    return options.get("params")
    

def get_models(large: bool, medium: bool, small: bool) -> list[str]:
    '''
    Returns
    -------
    "large" : ["qwen2.5:72b", "llama3.3:70b"]

    "medium" : ["qwen2.5:14b", "phi4:14b", "llava:13b"]

    "small" : ["llama3.1:8b", "mistral:7b", "qwen2.5:7b", "gemma:7b"]
    '''
    model_names = []
    if large:
        model_names.append("large")
    if medium:
        model_names.append("medium")
    if small:    
        model_names.append("small")
    models = get_config_content(models=model_names).get("models")
    return models


def get_ollama_params(isRational: bool) -> dict:
    '''
    Returns
    -------
    "mirostat": int

    "mirostat_eta": float

    "mirostat_tau": float

    "num_ctx": int

    "repeat_last_n": int

    "repeat_penalty": float

    "temperature": float

    "seed": long

    "num_predict": int

    "top_k": int

    "top_p": float

    "min_p": float
    '''
    if isRational:
        return get_config_content(ollama_param="rational").get("params")
    else:
        return get_config_content(ollama_param="creative").get("params")

    
def read_prompt(path: str) -> tuple[str, dict, str] | None:
    '''
    Parameters
    ----------
    path : str 
        Path to the prompt file

    Returns
    -------
    tuple : str, dict, str | None
        contains (instruction, data, tasks)
    '''
    data = read_json(path=path)
    if data:
        task, data, context, output = data.get("task"), data.get("data"), data.get("additional_context"), data.get("desired_output")
        print(f"[path={path}]\nTask: {task}\nData: {data}\nContext: {context}\nOutput: {output}\n")
        return task, data, context, output 
    else:
        return None
    

def read_all_prompts() -> list[tuple[str, str, dict, str]] | None:
    '''
    Returns
    -------
    list : tuple[str, str, dict, str] | None
        contains (path, instruction, data, tasks)
    '''
    prompts = []
    for root, dirs, files in os.walk('prompts'):
        for file in files:
            path = os.path.join(root, file)
            task, data, context, output = read_prompt(path)
            prompts.append((path, task, data, context, output))
    return prompts


if __name__ == "__main__":
    prompts = read_all_prompts()
    for prompt in prompts:
        path, task, data, context, output = prompt

    print(get_models(large=True, medium=True, small=True))
    print(get_ollama_params(isRational=True))
    print(get_ollama_params(isRational=False))