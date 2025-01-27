# Class for easy and standardized logging

import os
from datetime import datetime

# Make sure LOG folder exists
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log_prompt(prompt, response, execution_time, device_used):
    PROMPT_DIR = os.path.join(LOG_DIR, "prompt")
    os.makedirs(PROMPT_DIR, exist_ok=True)
    datetime = datetime.now()
    date = datetime.strftime("%d-%m-%Y")
    time = datetime.strftime("%H:%M Uhr (%Ss)")
    timestamp = datetime.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(PROMPT_DIR, f"prompt_{timestamp}.log")
    with open(log_filename, "w") as log_file:
                        log_file.write(f"\nDate: {date}  -  Time: {time}\n\n")
                        log_file.write(f"Device: {device_used}\n\n------------\n\n")
                        log_file.write(f"Prompt:\n{prompt}\n\n------------\n\n")
                        log_file.write(f"Answer:\n{response}\n\n------------\n\n")
                        log_file.write(f"Execution time: {execution_time} seconds\n\n")
