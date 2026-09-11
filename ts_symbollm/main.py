

import time

from . import logger
from .backends.ollama import OllamaBackend
from .config import config


####################################################################
### GENERATE PROMPTS
####################################################################

def generate_prompt(path: str, level: str) -> str:
    if level == "single":
        task, context, data, output = config.read_prompt(path=path)
        return f"Task:\n{task}\nData context:\n{context}\nData:\n{data}\nDesired output format:\n{output}\n"
    else:
        task, context, data, output = config.read_prompt(path=path)
        prompt = f"Task:\n{task}"
        for i in range(len(context)):
            string_context_extension = f"Data {i+1} context:\n{context[i]}\n"
            string_data_extension = f"Data {i+1}:\n{data[i]}\n"
            prompt = f"{prompt}{string_context_extension}{string_data_extension}"
        prompt = f"{prompt}\nDesired output format:\n{output}"
        print(f"[DEBUG] prompt\n{prompt}")
        return prompt

####################################################################
### MAIN
####################################################################

def main():
    backend = OllamaBackend()

    # Which model tiers to test — left off by default; callers can flip these on directly.
    models = []
    test_large_models, test_medium_models, test_small_models = False, False, False

    if test_large_models or test_medium_models or test_small_models:
        models = config.get_models(large=test_large_models, medium=test_medium_models, small=test_small_models)

    # Pull the models
    for model in models:
        backend.pull(model=model)
    print(f"[MAIN]: number of models = {len(models)}")

    params = config.get_ollama_parameter(isRational=True)
    
    # Cast to set, to remove duplicates, and back to get a list...
    prompt_paths = list(set(config.get_prompt_paths()))
    prompt_paths.sort()
    print(f"[MAIN]: number of prompts = {len(prompt_paths)}")

    counter = 1
    processing_counter = 1

    for model in models:
        print(f"\nTesting model: {model}\n")
        for path in prompt_paths:
            print(f"[COUTNER]: {counter}")
            logger.log_processing_order(path=path, model=model, counter=processing_counter)
            counter += 1
            start_time = time.time()
            description, level, data_representation = config.read_prompt_info(path=path)
            prompt = generate_prompt(path, level=level)
            response_string = backend.generate(model=model, prompt=prompt, params=params)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt_type=description, data_representation=data_representation, prompt=prompt, response=response_string, execution_time=execution_time)
            processing_counter += 1


def one_by_one_test():
    backend = OllamaBackend()
    models = config.get_models(large=True, medium=True, small=True)
    models.sort()
    params = [config.get_ollama_parameter(isRational=True), config.get_ollama_parameter(isRational=False)]
    prompt_paths = config.get_prompt_paths()
    prompt_paths.sort()

    counter = 1

    while(True):
        print(f"Continue ? (Y/N)\n")
        cont = input("  > ")
        if cont == "N" or cont == "n":
            break
        print(f"\n[RUN]: {counter}")
        print(f"\n[MODELS]:")
        for model in models:
            print(f"  --> {model}")
        print(f"\n[PARAMS]:")
        for param in ["rational", "none"]:
            print(f"  --> {param}")
        print(f"\n[PROMPTS]:")
        for i, path in enumerate(prompt_paths):
            print(f"  --> [{i}]:{path}")

        print(f"\n ############################################ \n")

        model = input("Enter the model to test: \n  > ")
        if model not in models:
            print("Invalid model name.")
            continue
        param = input("Enter the parameter to use: \n  > ")
        if param not in ["rational", "creative", "none"]:
            print("Invalid parameter name.")
            continue
        if param == "rational":
            param = params[0]
        elif param == "creative":
            param = params[1]
        else:
            param = None
        prompt_path = input("Enter the prompt path: \n  > ")
        if prompt_path in "0 1 2 3 4 5 6 7 8 9 10 11 12":
            prompt_path = prompt_paths[int(prompt_path)]
        elif prompt_path not in prompt_paths:
            print("Invalid prompt path.")
            continue

        start_time = time.time()
        description, level, data_representation = config.read_prompt_info(path=prompt_path)
        prompt = generate_prompt(prompt_path, level=level)
        response_string = backend.generate(model=model, prompt=prompt, params=param)
        execution_time = time.time() - start_time
        logger.log(model_name=model, ollama_params=param, prompt_path=path, prompt_type=description, data_representation=data_representation, prompt=prompt, response=response_string, execution_time=execution_time)
        print(f"[EXECUTION TIME]: {execution_time}")
        print(f"\n ######################################################### \n")
            

# Anfangstext "TS-SymboLLM", was kann er, was tut er, help?, etc.
# --help          - Aufzählung mit kurzer Beschreibung der einzelnen Funktionen
# --list %arg%    - Aufzählung der Elemente eines Ordners
#                   arg: models, datasets, prompts, params
# --reload %arg%  - Neues Laden eines Ordners
#                   arg: models, datasets, prompts, params
# --pull %args%   - Herunterladen eines in Ollama zur Verfügung stehenden LLMs
#                   args: Name/Bezeichner eines/mehrerer LMMs in Ollama
# --delete %args% - Löschen eines heruntergeladenen Ollama-Models
#                   args: Name/Bezeichner eines/mehrerer LMMs in Ollama
#
# --run %args%    - Start der Generation einer Antwort
#                   args: no-args, -m %model% -p %param% -ds %dataset% -pr %prompt%
# 
# 
# 
# 
# 
# 
#  