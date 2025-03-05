

import utils.funcllama as funcllama
import numpy as np
from config import config
from utils.helpers import generate_tsd, plot_tsd
import logger
import time
import os


def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False):
    if random:
        data = {"data": generate_tsd(kind_of_data="random", n_instances=np.random.randint(low=1, high=101), interval_sec=60)}
        config.write_json(data=data, path="prompts/random_test.json", overwrite=False)

    if temperature:
        data = {"data": generate_tsd(kind_of_data="temperature", n_instances=360, interval_sec=10)}
        config.write_json(data=data, path="prompts/temperature_test.json", overwrite=False)

    if sequence:
        for i in range(5):
            # 0: even numbers, 1: odd numbers, 2: squared numbers, 3: oscillating harmonic numbers, 4: prime numbers
            task = {"task": "Identify the underlying pattern of a numerical sequence where only every third value is provided. Fill in the missing numbers and return the complete sequence."}
            additional_context = {"additional_context": "The given data represents every third value of a complete numerical sequence. The missing values between the given numbers follow the same underlying pattern. Your goal is to infer the rule governing the sequence and reconstruct the full set of numbers."}
            desired_output = {"desired_output": "Return a list containing the complete numerical sequence, including the missing values in their correct positions. Do not include any additional commentary, only the reconstructed sequence. Also return a list containing the timestamps of every single value."}
            data = {"data": generate_tsd(kind_of_data="sequence", n_instances=10, interval_sec=3, sequence=i), **task, **additional_context, **desired_output}
            config.write_json(data=data, path=f"prompts/sequence_{['even', 'odd', 'squared', 'oscharm', 'prime'][i]}_test.json", overwrite=False)


def generate_prompt(path: str) -> str:
    task, data, context, output = config.read_prompt(path=path)
    return f"Task:\n{task}\nData:\n{data}\nAdditional context:\n{context}\nDesired output format:\n{output}\n"


def increase_logs_counter():
    logs = config.get_config_content(logs=True).get("logs")
    logs += 1
    config.write_json(data={"logs": logs}, path="src/config/config.json", overwrite=False)


def main():

    # Determine the models to test based on the user
    models = []
    if os.getlogin() == "dbisai":
        test_large_models, test_medium_models, test_small_models = True, True, True
    else:
        test_large_models, test_medium_models, test_small_models = False, False, False

    if test_large_models or test_medium_models or test_small_models:
        models = config.get_models(large=test_large_models, medium=test_medium_models, small=test_small_models)
    
    # Pull the models
    for model in models:
        funcllama.pull_ollama_model(model=model)

    params = config.get_ollama_params(isRational=True)
    
    prompt_paths, prompt_descriptions = config.get_all_prompt_paths_with_descriptions()

    for model in models:
        print(f"\nTesting model: {model}\n")
        for i in range(len(prompt_paths)):
            start_time = time.time()
            prompt_path = prompt_paths[i]
            description = prompt_descriptions[i]
            prompt = generate_prompt(prompt_path)
            response = funcllama.generate_response(model=model, prompt=prompt, params=params)
            response_string = funcllama.print_response(response)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt_type=description, prompt=prompt, response=response_string, execution_time=execution_time)
    increase_logs_counter()
            

if __name__ == "__main__":
    if os.getlogin() == "dbisai":
        main()
        #generate_data(random=False, temperature=False, sequence=True)
    else:
        generate_data(random=False, temperature=False, sequence=True)
        #funcllama.chat(model="qwen2.5:3b", params=config.get_ollama_params(isRational=True))