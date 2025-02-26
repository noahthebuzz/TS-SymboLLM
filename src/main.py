

import utils.funcllama as funcllama
import numpy as np
from config import config
from utils.helpers import generate_tsd, plot_tsd
import logger
import time


def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False):
    if random:
        data = {"data": generate_tsd(kind_of_data="random", n_instances=np.random.randint(low=1, high=101), interval_sec=60)}
        config.write_json(data=data, path="prompts/random_test.json", overwrite=False)

    if temperature:
        data = {"data": generate_tsd(kind_of_data="temperature", n_instances=360, interval_sec=10)}
        config.write_json(data=data, path="prompts/temperature_test.json", overwrite=False)

    if sequence:
        # 0: even numbers, 1: odd numbers, 2: squared numbers, 3: oscillating harmonic numbers, 4: prime numbers
        data = {"data": generate_tsd(kind_of_data="sequence", n_instances=10, interval_sec=3, sequence=0)}
        config.write_json(data=data, path="prompts/sequence_test.json", overwrite=False)


def generate_prompt(path: str) -> str:
    task, data, context, output = config.read_prompt(path=path)
    return f"Task:\n{task}\nData:\n{data}\nAdditional context:\n{context}\nDesired output format:\n{output}\n"


def increase_logs_counter():
    logs = config.get_config_content(logs=True).get("logs")
    logs += 1
    config.write_json(data={"logs": logs}, path="src/config/config.json", overwrite=False)


def main():
    test_large_models, test_medium_models, test_small_models = False, False, False
    models = []
    if test_large_models or test_medium_models or test_small_models:
        models = config.get_models(large=test_large_models, medium=test_medium_models, small=test_small_models)
    else:
        models = ["qwen2.5:14b", "qwen2.5:7b"]
    for model in models:
        funcllama.pull_ollama_model(model=model)

    params = config.get_ollama_params(isRational=True)

    for model in models:
        print(f"\nTesting model: {model}\n")
        for i in range(1):
            start_time = time.time()
            prompt_path = ["prompts/sequence_test.json", "prompts/temperature_test.json", "prompts/random_test.json"][i]
            prompt = generate_prompt(prompt_path)
            response = funcllama.generate_response(model=model, prompt=prompt, params=params)
            response_string = funcllama.print_response(response)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt=prompt, response=response_string, execution_time=execution_time)
    increase_logs_counter()
            

if __name__ == "__main__":
    generate_data(random=False, temperature=False, sequence=False)
    main()