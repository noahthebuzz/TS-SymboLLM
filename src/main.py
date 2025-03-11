

import utils.funcllama as funcllama
import numpy as np
from config import config
from utils.data_gen import generate_tsd, plot_tsd
import logger
import time
import os


def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False, multi_tsd: bool = False):
    if random:
        data = {"data": generate_tsd(kind_of_data="random", n_instances=np.random.randint(low=1, high=101), interval_sec=60)}
        config.write_prompt_json(data=data, path="prompts/random_test.json")

    if temperature:
        data = {"data": generate_tsd(kind_of_data="temperature", n_instances=360, interval_sec=10)}
        config.write_prompt_json(data=data, path="prompts/temperature_test.json")

    if sequence:
        for i in range(5):
            # 0: even numbers, 1: odd numbers, 2: squared numbers, 3: oscillating harmonic numbers, 4: prime numbers
            task = {"task": "Identify the underlying pattern of a numerical sequence where only every third value is provided. Fill in the missing numbers and return the complete sequence."}
            additional_context = {"additional_context": "The given data represents every third value of a complete numerical sequence. The missing values between the given numbers follow the same underlying pattern. Your goal is to infer the rule governing the sequence and reconstruct the full set of numbers."}
            desired_output = {"desired_output": "Return a list containing the complete numerical sequence, including the missing values in their correct positions. Do not include any additional commentary, only the reconstructed sequence. Also return a list containing the timestamps of every single value."}
            data = {"data": generate_tsd(kind_of_data="sequence", n_instances=10, interval_sec=3, sequence=i), **task, **additional_context, **desired_output}
            config.write_prompt_json(data=data, path=f"prompts/sequence_{['even', 'odd', 'squared', 'oscharm', 'prime'][i]}_test.json")

    if multi_tsd:
        # Temperature data and capacity data
        temp_data, cap_data = generate_tsd(kind_of_data="cpu_temp_and_cap", n_instances=(5*60), interval_sec=1)
        data = {"data1": temp_data, "data2": cap_data}
        config.write_prompt_json(data=data, path="prompts/multi/cpu_temp_and_cap_test.json")


def generate_prompt(path: str, level: str) -> tuple[str, str]:
    if level == "single":
        task, context, data, output = config.read_prompt(path=path)
        return f"Task:\n{task}\nData context:\n{context}\nData:\n{data}\nDesired output format:\n{output}\n"
    else:
        task, context_1, data_1, context_2, data_2, output = config.read_prompt(path=path)
        return f"Task:\n{task}\nData 1 context:\n{context_1}\nData 1:\n{data_1}\nData 2 context:\n{context_2}\nData 2:\n{data_2}\nDesired output format:\n{output}\n"


def increase_logs_counter():
    logs = config._get_config_content(logs=True).get("logs")
    logs += 1
    config.write_json(data={"logs": logs}, path="src/config/config.json")


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

    params = config.get_ollama_parameter(isRational=True)
    
    prompt_paths = config.get_prompt_paths()

    for model in models:
        print(f"\nTesting model: {model}\n")
        for path in prompt_paths:
            start_time = time.time()
            description, level = config.read_prompt_description_and_level(path=path)
            prompt = generate_prompt(path, level=level)
            response = funcllama.generate_response(model=model, prompt=prompt, params=params)
            response_string = funcllama.print_response(response)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt_type=description, prompt=prompt, response=response_string, execution_time=execution_time)
    increase_logs_counter()
            

if __name__ == "__main__":
    if os.getlogin() == "dbisai":
        generate_data(random=False, temperature=False, sequence=False, multi_tsd=True)
        main(test_multi=True)
        #config.read_all_prompts()
    else:
        #generate_data(random=False, temperature=False, sequence=False, multi_tsd=True)
        #funcllama.chat(model="qwen2.5:3b", params=config.get_ollama_params(isRational=True))
        paths = config.get_prompt_paths()
        for path in paths:
            print(f"[DEBUG]: {path}")
            config.read_prompt(path=path)