import json
import os


####################################################################
### READ AND WRITE JSON FILES
####################################################################

def write_prompt_json(data: dict, path: str) -> bool:
    try:
        new_data = {}
        prompt = {}
        prompt_info = {}
        if os.path.exists(path):
            old_data = read_json(path=path)
            if old_data:
                prompt, prompt_info = old_data.get("prompt"), old_data.get("prompt_info")
                for key in prompt.keys():
                    if key not in data.keys():
                        data[key] = prompt[key]
        new_data = {"prompt": data, "prompt_info": prompt_info}
        with open(path, 'w') as file:
            json.dump(new_data, file, indent=4)
        return True
    except Exception as e:
        return False


def write_json(data: dict, path: str, overwrite: bool = False) -> bool:
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
    


####################################################################
### READ CONFIGURATION FILE
####################################################################
    
def _get_config_content(setup_usr: str = None, logs: bool = None, ollama_param: str = None, models: list[str] = None) -> dict | None:
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
    

def get_user_setup() -> dict:
    '''
    Returns the PC setup for the current user
    '''
    usr = os.getlogin()
    return _get_config_content(setup_usr=usr).get(usr)


def get_logs() -> int:
    '''
    Returns the current log counter
    '''
    return _get_config_content(logs=True).get("logs")
    

def get_ollama_parameter(isRational: bool) -> dict:
    '''
    Returns the parameters for the OLLAMA model
    '''
    options = _get_config_content(ollama_param="rational" if isRational else "creative")
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
    models = _get_config_content(models=model_names).get("models")
    return models


####################################################################
### READ PROMPT FILES
####################################################################

def read_prompt(path: str) -> tuple[str, str, str, dict, str] | tuple[str, str, str, dict, str, dict, str] | None:
    '''
    Reads the prompt file in the specified path.

    Returns the description, task, context_n, data_n, and desired output.
    '''
    data = read_json(path=path)
    #print(f"[CONFIG]: {data}")
    prompt_data = data.get("prompt")
    prompt_info = data.get("prompt_info")

    level = prompt_info.get("level")

    if level == "single":
        #desc = prompt_info.get("description")
        task, context, data, output = prompt_data.get("task"), prompt_data.get("data_context"), prompt_data.get("data"), prompt_data.get("desired_output")
        print(f"\n[READING]\n[path={path}]\nTask: {task}\nContext: {context}\nData: {data}\nDesired Output: {output}\n")
        return task, context, data, output
    
    # TODO dynamically get the number of data contexts and data
    elif level == "multi":
        #desc = prompt_info.get("description")
        task, context_1, data_1, context_2, data_2, output = prompt_data.get("task"), prompt_data.get("data_context_1"), prompt_data.get("data_1"), prompt_data.get("data_context_2"), prompt_data.get("data_2"), prompt_data.get("desired_output")
        print(f"\n[READING]\n[path={path}]\nTask: {task}\nContext [1]: {context_1}\nData [1]: {data_1}\nContext [2]: {context_2}\nData [2]: {data_2}\nDesired Output: {output}\n")
        return task, context_1, data_1, context_2, data_2, output
    else:
        return None
    

def get_data_from_prompt(path: str) -> dict:
    '''
    Reads the prompt file in the specified path.

    Returns the data from the prompt.
    '''
    a, level, b = read_prompt_info(path=path)
    if level == "single":
        task, context, data, output = read_prompt(path=path)
        return data
    elif level == "multi":
        prompt = read_prompt(path=path)
        return prompt[2:(len(prompt)-1):2]
    else:
        return None
    

def read_prompt_info(path: str) -> tuple[str, str, str]:
    '''
    Reads the prompt file in the specified path.

    Returns the description and level of the prompt.
    '''
    print(f"\n[READING INFO]\n[path={path}]\n")
    data = read_json(path=path)
    prompt_info = data.get("prompt_info")
    return prompt_info.get("description"), prompt_info.get("level"), prompt_info.get("representation")


def get_prompt_paths(path: str = "prompts"):
    prompts_paths = []

    for root, dirs, files in os.walk(path):
        #print(f"[CONFIG]: {root}, {dirs}, {files}")
        for file in files:
            #print(f"[CONFIG]: Found {file} in {path}")
            if file.endswith(".json"):
                prompts_paths.append(os.path.join(root, file))
        
        for dir in dirs:
            prompts_paths.extend(get_prompt_paths(os.path.join(root, dir)))
    return list(set(prompts_paths))