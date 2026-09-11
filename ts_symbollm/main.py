

import numpy as np
import time
from datetime import datetime

from . import logger
from .config import config
from .utils import datagen
from .utils import funcllama


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
    raw_data, rounded_data, paasax_data = {"data": datasets[0]}, {"data": datasets[1]}, {"data": datasets[2]}

    # Plot data
    datagen.plot_single_tsd(data=raw_data["data"], abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=raw_path)
    datagen.plot_single_tsd(data=rounded_data["data"], abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=rounded_path)
    datagen.plot_single_tsd(data=paasax_data["data"], abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C)", save=True, location=paasax_path)

    paasax_string_data = {"data": datagen.get_sax_string(paasax_data)}
    #print(f"sax data:\n{paasax_data}")
    #print(f"sax string data:\n{paasax_string_data}")

    # Write data to json files
    config.write_prompt_json(data=raw_data    , path=raw_path + filename + "_test.json")
    config.write_prompt_json(data=rounded_data, path=rounded_path + filename + "_test.json")
    config.write_prompt_json(data=paasax_string_data, path=paasax_path + filename + "_test.json")


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
    paasax_data = [cap_data[2], temp_data[2]]

    paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    filename = "cpu_temp_and_cap"
    labels = ["Capacity (%)", "Temperature (°C)"]

    # Plot data
    datagen.plot_multi_tsd(data=raw_data, labels=labels, abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[0])
    datagen.plot_multi_tsd(data=rounded_data, labels=labels, abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[1])
    datagen.plot_multi_tsd(data=paasax_data, labels=labels, abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%)", save=True, location=paths[2])

    sax_string = [datagen.get_sax_string({"data": paasax_data[0]}), datagen.get_sax_string({"data": paasax_data[1]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0]}, path=f"{paths[0]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1]}, path=f"{paths[1]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0]}, path=f"{paths[2]}{filename}_test.json")


def _generate_multi_3_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data = [cap_data[0], temp_data[0], rpm_data[0]]
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3]]
    #rounded_data = [cap_data[1], temp_data[1], rpm_data[1]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2]]

    #paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_3"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)"]

    # Plot data
    datagen.plot_multi_tsd(data=raw_data_plot, labels=labels, abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM)", save=True, location=paths[0])
    #datagen.plot_multi_tsd(data=rounded_data, labels=labels, abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM)", save=True, location=paths[1])
    datagen.plot_multi_tsd(data=paasax_data, labels=labels, abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM)", save=True, location=paths[1])

    sax_string = [datagen.get_sax_string({"data": paasax_data[0]}), datagen.get_sax_string({"data": paasax_data[1]}), datagen.get_sax_string({"data": paasax_data[2]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0]}, path=f"{paths[0]}{filename}_test.json")
    #config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1], "data_3": rpm_data[1]}, path=f"{paths[1]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2]}, path=f"{paths[1]}{filename}_test.json")


def _generate_multi_4_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    volt_data = datagen.generate_voltage_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data = [cap_data[0], temp_data[0], rpm_data[0], volt_data[0]]
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3], volt_data[3]]
    #rounded_data = [cap_data[1], temp_data[1], rpm_data[1], volt_data[1]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2], volt_data[2]]

    #paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_4"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)", "Voltage (%)"]

    # Plot data
    datagen.plot_multi_tsd(data=raw_data_plot, labels=labels, abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%)", save=True, location=paths[0])
    #datagen.plot_multi_tsd(data=rounded_data, labels=labels, abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%)", save=True, location=paths[1])
    datagen.plot_multi_tsd(data=paasax_data, labels=labels, abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%)", save=True, location=paths[1])

    sax_string = [datagen.get_sax_string({"data": paasax_data[0]}), datagen.get_sax_string({"data": paasax_data[1]}), datagen.get_sax_string({"data": paasax_data[2]}), datagen.get_sax_string({"data": paasax_data[3]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0], "data_4": volt_data[0]}, path=f"{paths[0]}{filename}_test.json")
    #config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1], "data_3": rpm_data[1], "data_4": volt_data[1]}, path=f"{paths[1]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2], "data_4": sax_string[3]}, path=f"{paths[1]}{filename}_test.json")


def _generate_multi_5_data():
    n_instances, interval_sec, time = 300, 1, datetime.now()
    temp_data = datagen.generate_temperature_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    cap_data = datagen.generate_capacity_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    rpm_data = datagen.generate_fanspeed_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    volt_data = datagen.generate_voltage_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    roomtemp_data = datagen.generate_roomtemp_tsd(n_instances=n_instances, interval_sec=interval_sec, time=time)
    raw_data = [cap_data[0], temp_data[0], rpm_data[0], volt_data[0], roomtemp_data[0]]
    raw_data_plot = [cap_data[0], temp_data[0], rpm_data[3], volt_data[3], roomtemp_data[0]]
    #rounded_data = [cap_data[1], temp_data[1], rpm_data[1], volt_data[1], roomtemp_data[1]]
    paasax_data = [cap_data[2], temp_data[2], rpm_data[2], volt_data[2], roomtemp_data[2]]

    #paths = ["prompts/multi/float/", "prompts/multi/integer/", "prompts/multi/paasax/"]
    paths = ["prompts/multi/float/", "prompts/multi/paasax/"]
    filename = "multi_5"
    labels = ["Capacity (%)", "Temperature (°C)", "RPM (X*100)", "Voltage (%)", "Room Temperature (°C)"]

    # Plot data
    datagen.plot_multi_tsd(data=raw_data_plot, labels=labels, abstraction_level="raw", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%) / Room Temperature (°C)", save=True, location=paths[0])
    #datagen.plot_multi_tsd(data=rounded_data, labels=labels, abstraction_level="rounded", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%) / Room Temperature (°C)", save=True, location=paths[1])
    datagen.plot_multi_tsd(data=paasax_data, labels=labels, abstraction_level="paasax", title=filename, xlabel="Time (HH:MM:SS)", ylabel="Temperature (°C) / Capacity (%) / Fanspeed (RPM) / Voltage (%) / Room Temperature (°C)", save=True, location=paths[1])

    sax_string = [datagen.get_sax_string({"data": paasax_data[0]}), datagen.get_sax_string({"data": paasax_data[1]}), datagen.get_sax_string({"data": paasax_data[2]}), datagen.get_sax_string({"data": paasax_data[3]}), datagen.get_sax_string({"data": paasax_data[4]})]

    # Write data to json files
    config.write_prompt_json(data={"data_1": temp_data[0], "data_2": cap_data[0], "data_3": rpm_data[0], "data_4": volt_data[0], "data_5": roomtemp_data[0]}, path=f"{paths[0]}{filename}_test.json")
    #config.write_prompt_json(data={"data_1": temp_data[1], "data_2": cap_data[1], "data_3": rpm_data[1], "data_4": volt_data[1], "data_5": roomtemp_data[1]}, path=f"{paths[1]}{filename}_test.json")
    config.write_prompt_json(data={"data_1": sax_string[1], "data_2": sax_string[0], "data_3": sax_string[2], "data_4": sax_string[3], "data_5": sax_string[4]}, path=f"{paths[1]}{filename}_test.json")

###########################################################

def generate_data(random: bool = False, temperature: bool = False, sequence: bool = False, cpu_temp_cap_tsd: bool = False, multi_3_tsd: bool = False, multi_4_tsd: bool = False, multi_5_tsd: bool = False):
    if random:
        _generate_random_data()

    if temperature:
        _generate_temperature_data()

    if sequence:
        _generate_sequence_data()

    if cpu_temp_cap_tsd:
        _generate_multi_cpu_data()

    if multi_3_tsd:
        _generate_multi_3_data()

    if multi_4_tsd:
        _generate_multi_4_data()

    if multi_5_tsd:
        _generate_multi_5_data()

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

    # Which model tiers to test — left off by default; callers can flip these on directly.
    models = []
    test_large_models, test_medium_models, test_small_models = False, False, False

    if test_large_models or test_medium_models or test_small_models:
        models = config.get_models(large=test_large_models, medium=test_medium_models, small=test_small_models)
    
    # Pull the models
    for model in models:
        funcllama.pull_ollama_model(model=model)
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
            response = funcllama.generate_response(model=model, prompt=prompt, params=params)
            response_string = funcllama.print_response(response)
            execution_time = time.time() - start_time
            logger.log(model_name=model, ollama_params=params, prompt_type=description, data_representation=data_representation, prompt=prompt, response=response_string, execution_time=execution_time)
            processing_counter += 1


def one_by_one_test():
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
        response = funcllama.generate_response(model=model, prompt=prompt, params=param)
        response_string = funcllama.print_response(response)
        execution_time = time.time() - start_time
        logger.log(model_name=model, ollama_params=param, prompt_path=path, prompt_type=description, data_representation=data_representation, prompt=prompt, response=response_string, execution_time=execution_time)
        print(f"[EXECUTION TIME]: {execution_time}")
        print(f"\n ######################################################### \n")
            

if __name__ == "__main__":
    #generate_data(random=True, temperature=True, sequence=True, multi_tsd=True)
    #main()
    #one_by_one_test()
    generate_data(temperature=True)


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