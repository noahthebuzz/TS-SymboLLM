# Class for easy and standardized logging
import os
from datetime import datetime
from config import config

def format_logs_counter(counter: int) -> str:
        counter = str(counter)
        return "0" * (6 - len(counter)) + counter

def log(usr: str, model_name:str, model_setup: dict, prompt_type:str, prompt_nr: int, prompt: str, response: str, execution_time: float, test: bool = False):
        
        # Fetch config data
        config_data = config.read_file()
        logs_counter = config_data["logs"]
        setup = config_data["setup"][usr]

        # Ensure the logs directory exists
        if test:
               logs_counter_string = "test"
        else:
                logs_counter_string = format_logs_counter(logs_counter)
        dirs = f"logs/{usr}/{logs_counter_string}"
        os.makedirs(dirs, exist_ok=True)

        # Write to log file
        #print(f"\n[INFO]: Writing to log file...")
        time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"./{dirs}/{prompt_type}_{time_id}.log"

        timestamp = datetime.now().strftime("%H:%M:%S   %d-%m-%Y")

        with open(log_filename, "w") as log_file:
                log_file.write(f"Timestamp: HH-MM-SS   DD-MM-YYYY\n")
                log_file.write(f"           {timestamp}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Model: {model_name}\n")
                log_file.write(f"--------------------------\n")
                if model_setup is not None:
                        log_file.write(f"     mirostat      : {model_setup['mirostat']}\n")
                        log_file.write(f"     mirostat_eta  : {model_setup['mirostat_eta']}\n")
                        log_file.write(f"     mirostat_tau  : {model_setup['mirostat_tau']}\n")
                        log_file.write(f"     num_ctx       : {model_setup['num_ctx']}\n")
                        log_file.write(f"     repeat_last_n : {model_setup['repeat_last_n']}\n")
                        log_file.write(f"     repeat_penalty: {model_setup['repeat_penalty']}\n")
                        log_file.write(f"     temperature   : {model_setup['temperature']}\n")
                        log_file.write(f"     seed          : {model_setup['seed']}\n")
                        log_file.write(f"     num_predict   : {model_setup['num_predict']}\n")
                        log_file.write(f"     top_k         : {model_setup['top_k']}\n")
                        log_file.write(f"     top_p         : {model_setup['top_p']}\n")
                        log_file.write(f"     min_p         : {model_setup['min_p']}\n\n")
                        log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Processor: {setup['Processor']}\n")
                log_file.write(f"RAM: {setup['RAM']}\n")
                log_file.write(f"GPU: {setup['GPU']}\n")
                log_file.write(f"OS: {setup['OS']}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Prompt [{prompt_nr}]:\n{prompt}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Answer:\n{response}\n\n")
                log_file.write(f"-------------------------------------------\n\n")
                log_file.write(f"Execution time: {execution_time} seconds\n\n")
        
        #print(f"\n[INFO]: Log file written successfully: {log_filename}")

        # Update logs counter
        if not test:
            logs_counter += 1
            config.write_file({"logs": logs_counter})


if __name__ == "__main__":
        log(usr="noahthebuzz", model_name="TestPT", model_setup=None, prompt_type="logic", prompt_nr=1, prompt="Levi is my father, Grisha is Levi's brother and Eren is Grisha's son. Who is Eren to me ?", response="Eren is my cousin", execution_time=0.5, test=True)
