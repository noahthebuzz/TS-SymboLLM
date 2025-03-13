

import utils.funcllama as funcllama
import numpy as np
from config import config
from utils import datagen
import logger
import time
import os
from datetime import datetime


####################################################################
### GENERATE DATA
####################################################################

def _generate_random_data():
    raw_path = "prompts/single/float/"
    rounded_path = "prompts/single/integer/"
    filename = "random"
    datasets = datagen.generate_random_tsd(n_instances=np.random.randint(low=1, high=101), interval_sec=60)
    raw_data, rounded_data = {"data": datasets[0]}, {"data": datasets[1]}
    
    # Plot data
    datagen.plot_single_tsd(data=raw_data["data"], abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Value", save=True, location=raw_path)
    datagen.plot_single_tsd(data=rounded_data["data"], abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Value", save=True, location=rounded_path)

    # Write data to json files
    config.write_prompt_json(data=raw_data    , path=raw_path + filename + "_test.json")
    config.write_prompt_json(data=rounded_data, path=rounded_path + filename + "_test.json")


def _generate_temperature_data():
    raw_path = "prompts/single/float/"
    rounded_path = "prompts/single/integer/"
    paasax_path = "prompts/single/paasax/"
    filename = "temperature"
    datasets = datagen.generate_temperature_tsd(n_instances=300, interval_sec=1)
    raw_data, rounded_data, paasax_data = {"data": datasets[0]}, {"data": datasets[1]}, {}

    # Plot data
    datagen.plot_single_tsd(data=raw_data["data"], abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=raw_path)
    datagen.plot_single_tsd(data=rounded_data["data"], abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=rounded_path)
    #datagen.plot_single_tsd(data=paasax_data["data"], abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=paasax_path)

    # Write data to json files
    config.write_prompt_json(data=raw_data    , path=raw_path + filename + "_test.json")
    config.write_prompt_json(data=rounded_data, path=rounded_path + filename + "_test.json")
    #config.write_prompt_json(data=paasax_data, path=paasax_path + filename + "_test.json")


def _generate_sequence_data():
    dir, name = "prompts/single/", "sequence_"
    # 0: even numbers, 1: odd numbers, 2: squared numbers, 3: oscillating harmonic numbers, 4: prime numbers
    sequences = ['even', 'odd', 'squared', 'harmosc', 'prime']
    paths = ['integer/', 'integer/', 'integer/', 'float/', 'integer/']
    path = ""
    for i in range(5):
        filename = f"{name}{sequences[i]}"
        path = f"{dir}{paths[i]}{filename}_test.json"
        datasets = datagen.generate_sequence_tsd(n_instances=10, interval_sec=3, sequence=i)
        data = {"data": datasets}
        # TODO plot data -> filename
        # Plot data
        datagen.plot_single_tsd(data=data["data"], abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel=f"{sequences[i].capitalize()} Values", save=True, location=dir + paths[i])

        # Write data to json files
        config.write_prompt_json(data=data, path=path)


def _generate_multi_cpu_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data = [cap_data[0], temp_data[0]]
    rounded_data = [cap_data[1], temp_data[1]]
    #paasax_data = [cap_data[2], temp_data[2]]
    paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    filename = "cpu_temp_and_cap"
    labels = ["Capacity (%)", "Temperature (°C)"]
    # Plot data
    datagen.plot_multi_tsd(data=raw_data, labels=labels, abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[0])
    datagen.plot_multi_tsd(data=rounded_data, labels=labels, abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[1])
    #datagen.plot_multi_tsd(data=paasax_data, abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[2])

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1]}, path=f"{paths[1]}{filename}_test.json")
    #config.write_prompt_json(data={"data_1": temp_data[2], "data_2": cap_data[2]}, path=f"{paths[2]}{filename}_test.json")
    

def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False, multi_tsd: bool = False):
    if random:
        _generate_random_data()

    if temperature:
        _generate_temperature_data()

    if sequence:
        _generate_sequence_data()

    if multi_tsd:
        _generate_multi_cpu_data()

####################################################################
### GENERATE PROMPTS
####################################################################

def generate_prompt(path: str, level: str) -> str:
    if level == "single":
        task, context, data, output = config.read_prompt(path=path)
        return f"Task:\n{task}\nData context:\n{context}\nData:\n{data}\nDesired output format:\n{output}\n"
    else:
        task, context_1, data_1, context_2, data_2, output = config.read_prompt(path=path)
        return f"Task:\n{task}\nData 1 context:\n{context_1}\nData 1:\n{data_1}\nData 2 context:\n{context_2}\nData 2:\n{data_2}\nDesired output format:\n{output}\n"

####################################################################
### INCREASE LOGS COUNTER
####################################################################

def increase_logs_counter():
    logs = config._get_config_content(logs=True).get("logs")
    logs += 1
    config.write_json(data={"logs": logs}, path="src/config/config.json")

####################################################################
### MAIN
####################################################################

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
            description, level, data_representation = config.read_prompt_info(path=path)
            prompt = generate_prompt(path, level=level)
            response = funcllama.generate_response(model=model, prompt=prompt, params=params)
            response_string = funcllama.print_response(response)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt_type=description, data_representation=data_representation, prompt=prompt, response=response_string, execution_time=execution_time)
    increase_logs_counter()
            

if __name__ == "__main__":
    if os.getlogin() == "dbisai":
        generate_data(random=False, temperature=False, sequence=False, multi_tsd=True)
        main(test_multi=True)
        #config.read_all_prompts()
    else:
        generate_data(random=True, temperature=True, sequence=True, multi_tsd=True)
        '''paths = config.get_prompt_paths()
        for path in paths:
            print(f"[DEBUG]: {path}")
            config.read_prompt(path=path)'''