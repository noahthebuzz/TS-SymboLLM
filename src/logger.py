import os
from datetime import datetime
from config import config


def determine_usr() -> str:
    usr = os.getlogin()
    return usr


def format_logs_counter(counter: int) -> str:
    counter = str(counter)
    return '0' * (6 - len(counter)) + counter


def log(model_name: str, ollama_params: dict, prompt_type: str, data_representation: str, prompt: str, response: str, execution_time: float, test: bool = False):
    usr = determine_usr()
    config_data = config._get_config_content(setup_usr=usr, logs=True)

    if config_data is None:
        usr_setup, logs = None, 0
    else:
        if test:
            usr_setup, logs = config_data.get(usr), -1
        else:
            usr_setup, logs = config_data.get(usr), config_data.get("logs")
        
    logs_string = format_logs_counter(logs)

    large_models = config.get_models(True, False, False)
    medium_models = config.get_models(False, True, False)
    small_models = config.get_models(False, False, True)

    if model_name in large_models:
         model_size = "large"
    elif model_name in medium_models:
         model_size = "medium"
    elif model_name in small_models:
         model_size = "small"

    dirs = f"logs/{usr}/{prompt_type}/{model_size}/{model_name}/{logs_string}"
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
                if usr_setup is not None:
                    log_file.write(f"Processor: {usr_setup['Processor']}\n")
                    log_file.write(f"RAM: {usr_setup['RAM']}\n")
                    log_file.write(f"GPU: {usr_setup['GPU']}\n")
                    log_file.write(f"OS: {usr_setup['OS']}\n\n")
                    log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Representation/Form of Data: {data_representation}\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Prompt:\n{prompt}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Answer:\n{response}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Execution time: {execution_time} seconds\n\n")
    
    # Update logs counter
    #if not test:
    #    logs += 1
    #    config.write_json(data={"logs": logs}, path="src/config/config.json", overwrite=False)

if __name__ == "__main__":
     print(determine_usr())