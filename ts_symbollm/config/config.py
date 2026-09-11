import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
USER_CONFIG_ENV_VAR = "TS_SYMBOLLM_CONFIG"
DEFAULT_USER_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "ts-symbollm", "config.json")


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

def _user_config_path() -> str | None:
    '''
    Resolves the user's config override file, if any: the path from the
    TS_SYMBOLLM_CONFIG environment variable takes precedence, otherwise
    ~/.config/ts-symbollm/config.json is used if it exists.
    '''
    override = os.environ.get(USER_CONFIG_ENV_VAR)
    if override:
        return override
    if os.path.exists(DEFAULT_USER_CONFIG_PATH):
        return DEFAULT_USER_CONFIG_PATH
    return None


def _load_config() -> dict:
    '''
    Loads the shipped default config and overlays a user config file, if one
    is found. Overriding a top-level section (e.g. "models") replaces that
    section entirely rather than merging its individual keys.
    '''
    data = read_json(CONFIG_PATH) or {}
    user_path = _user_config_path()
    if user_path:
        user_data = read_json(user_path)
        if user_data:
            data.update(user_data)
    return data


def get_ollama_parameter(isRational: bool) -> dict:
    '''
    Returns the parameters for the OLLAMA model
    '''
    key = "rational" if isRational else "creative"
    return _load_config().get("ollama_parameter", {}).get(key)


def get_models(large: bool, medium: bool, small: bool) -> list[str]:
    '''
    Returns
    -------
    "large" : ["qwen2.5:72b", "gemma3:27b", "mistral-small:24b"]

    "medium" : ["qwen2.5:14b", "phi4:14b"]

    "small" : ["mistral:7b", "qwen2.5:7b", "gemma:7b"]
    '''
    tiers = _load_config().get("models", {})
    model_names = []
    if large:
        model_names.extend(tiers.get("large", []))
    if medium:
        model_names.extend(tiers.get("medium", []))
    if small:
        model_names.extend(tiers.get("small", []))
    return model_names


def resolve_model_size(model: str) -> str:
    '''
    Returns which configured tier ("large", "medium", "small") a model
    name belongs to, or "unknown" if it isn't listed in any tier.
    '''
    tiers = _load_config().get("models", {})
    for size, names in tiers.items():
        if model in names:
            return size
    return "unknown"


def get_representation_defaults() -> dict:
    '''
    Returns the default representation settings: rounding decimal places,
    the SAX alphabet size ("levels"), and the symbolic segment-length
    divisor used to derive a default number of PAA segments from a series'
    length (n / symbolic_segment_length).
    '''
    return _load_config().get("representation", {})


def get_plotting_config() -> dict:
    '''
    Returns the default plotting settings (diagram output directory).
    '''
    return _load_config().get("plotting", {})


####################################################################
### READ PROMPT FILES
####################################################################

def read_prompt(path: str) -> tuple[str, str, dict, str] | tuple[str, list[str], list[dict], str] | None:
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
        number_of_datasets = prompt_info.get("datasets")
        print(f"[DEBUG] number of datasets: {number_of_datasets}")
        #desc = prompt_info.get("description")
        task = prompt_data.get("task")
        output = prompt_data.get("desired_output")
        context, data = [], []
        for i in range(1, number_of_datasets + 1):
            context.append(prompt_data.get(f"data_context_{i}"))
            data.append(prompt_data.get(f"data_{i}"))
        return task, context, data, output
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