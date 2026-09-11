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