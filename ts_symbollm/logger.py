import os
from datetime import datetime

from .config import config


def log(model_name: str, ollama_params: dict, prompt_path: str, prompt_type: str, data_representation: str, prompt: str, response: str, execution_time: float):
    model_size = config.resolve_model_size(model_name)

    params = ""
    if ollama_params is None:
        params = "default"
    else:
        params = "rational"

    dirs = f"logs/{prompt_type}/{model_size}/{model_name}/{params}/{data_representation}"
    os.makedirs(dirs, exist_ok=True)

    time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_filename = f"./{dirs}/{time}.log"

    timestamp = datetime.now().strftime("%H:%M:%S   %d-%m-%Y")

    with open(log_filename, "w") as log_file:
                log_file.write(f"Timestamp: HH-MM-SS   DD-MM-YYYY\n")
                log_file.write(f"           {timestamp}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Model: {model_name}\n")
                log_file.write(f"--------------------------\n")
                if ollama_params is not None:
                        log_file.write(f"     mirostat      : {ollama_params['mirostat']}\n")
                        log_file.write(f"     mirostat_eta  : {ollama_params['mirostat_eta']}\n")
                        log_file.write(f"     mirostat_tau  : {ollama_params['mirostat_tau']}\n")
                        log_file.write(f"     num_ctx       : {ollama_params['num_ctx']}\n")
                        log_file.write(f"     repeat_last_n : {ollama_params['repeat_last_n']}\n")
                        log_file.write(f"     repeat_penalty: {ollama_params['repeat_penalty']}\n")
                        log_file.write(f"     temperature   : {ollama_params['temperature']}\n")
                        log_file.write(f"     seed          : {ollama_params['seed']}\n")
                        log_file.write(f"     num_predict   : {ollama_params['num_predict']}\n")
                        log_file.write(f"     top_k         : {ollama_params['top_k']}\n")
                        log_file.write(f"     top_p         : {ollama_params['top_p']}\n")
                        log_file.write(f"     min_p         : {ollama_params['min_p']}\n\n")
                        log_file.write(f"-------------------------------------------\n\n")
                else:
                     log_file.write(f"Default\n")
                     log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Path: {prompt_path}\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Prompt: {data_representation}\n{prompt}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Answer:\n{response}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Execution time: {execution_time} seconds\n\n")


def log_processing_order(path: str, model: str, counter: int):
    data = {f"{counter}": {"path": path, "model": model}}
    config.write_json(data=data, path="logs/processing_order.json", overwrite=False)