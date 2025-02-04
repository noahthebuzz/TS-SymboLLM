# Class for easy and standardized logging

import os
from datetime import datetime
from src.config import config

def format_logs_counter(counter: int) -> str:
        counter = str(counter)
        return "0" * (6 - len(counter)) + counter

def log(usr: str, model_name:str, prompt_type:str, prompt: str, response: str, execution_time: float, test: bool = False):
        
        # Fetch config data
        config_data = config.read_file()
        logs_counter = config_data["logs"]
        setup = config_data["setup"][usr]

        # Ensure the logs directory exists
        logs_counter_string = format_logs_counter(logs_counter)
        dirs = f"logs/{usr}/{logs_counter_string}"
        os.makedirs(dirs, exist_ok=True)

        # Write to log file
        #print(f"\n[INFO]: Writing to log file...")
        time_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"./{dirs}/{prompt_type}_{time_id}.log"

        timestamp = datetime.now().strftime("%H:%M:%S   %d-%m-%Y")

        with open(log_filename, "w") as log_file:
                log_file.write(f"Timestamp: HH-MM-SS   DD-MM-YYYY")
                log_file.write(f"           {timestamp}\n\n-------------------------------------------\n\n")
                log_file.write(f"Model: {model_name}\n\n-------------------------------------------\n\n")
                log_file.write(f"Processor: {setup['Processor']}\n")
                log_file.write(f"RAM: {setup['RAM']}\n")
                log_file.write(f"GPU: {setup['GPU']}\n")
                log_file.write(f"OS: {setup['OS']}\n\n-------------------------------------------\n\n")
                log_file.write(f"Prompt:\n{prompt}\n\n-------------------------------------------\n\n")
                log_file.write(f"Answer:\n{response}\n\n-------------------------------------------\n\n")
                log_file.write(f"Execution time: {execution_time} seconds\n\n")
        
        #print(f"\n[INFO]: Log file written successfully: {log_filename}")

        # Update logs counter
        if not test:
            logs_counter += 1
            config.write_file({"logs": logs_counter})


if __name__ == "__main__":
        log("noahthebuzz", "TestPT", "Logic", "Levi is my father, Grisha is Levi's brother and Eren is Grisha's son. Who is Eren to me ?", "Eren is my son", 0.5, test=True)
