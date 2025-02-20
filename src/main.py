#from ollama import chat, ChatResponse, list, GenerateResponse, generate, pull
import ollama

'''
EXAMPLES
'''

'''response: GenerateResponse = generate(
    model='qwen2.5:14b', 
    prompt='Ich gebe dir 10 Temperaturdaten von einem Prozessor, die immer im Abstand von 10 Sekunden gemessen wurden. Gib mir eine Interpretation der Daten: 50 45 43 46 52 55 50 60 75 92',
    options={
        'mirostat': 2,
        'seed': 0,
        'temperature': 0,
        'top_k': 0,
        'top_p': 0.1,
        'min_p': 0.0,
        'num_ctx': 2048
    }
) 
print(f"\n[GENERATE]:\n{response['response']}")'''

'''
response: ChatResponse = chat(
    model='qwen2.5:14b', 
    messages=[
        {
            'role': 'user',
            'content': 'Ich gebe dir 10 Temperaturdaten von einem Prozessor, die immer im Abstand von 10 Sekunden gemessen wurden. Gib mir eine Interpretation der Daten: 50 45 43 46 52 55 50 60 75 92'
        }
    ],
    options={
        'mirostat': 0,
        'seed': 0,
        'temperature': 0,
        'top_k': 0,
        'top_p': 0.1,
        'min_p': 0.0,
        'num_ctx': 2048
    }
)
print(f"\n[CHAT]:\n{response['message']['content']}")
'''




from utils import helpers
from config import config
from typing import Iterator
import ollama
from ollama import GenerateResponse

def load_ollama_parameter(isRational: bool) -> dict:
    options = config.get_content(ollama_param=isRational)
    return options['param']


def generate_temperature_data_file(n_data_points: int = 300) -> dict:
    params = {
        "stable_temp": 55,
        "stable_deviation": 0.75,
        "unstable_start": round(n_data_points // 2),
        "unstable_deviation": 2.5,
        "increase_start": round(n_data_points // 1.35),
        "final_temp": 90
    }
    print(params)
    datalist = helpers.generate_temperature_data(n_data_points=n_data_points, params=params)
    config.write_file("list", datalist, "prompts/error_tsd/test_data.json")
    return datalist


def pull_ollama_model(model: str) -> None:
    response = ollama.pull(model=model, stream=True)
    progress_states = set()
    for progress in response:
        if progress.get('status') in progress_states:
            continue
        progress_states.add(progress.get('status'))
        print(progress.get('status'))
    print('\n')


def pull_models(large: bool = False, medium: bool = False, small: bool = False) -> tuple:
    ring = 0
    large_models = []
    medium_models = []
    small_models = []
    with open("src/config/models.txt", "r", encoding="utf-8") as f:
        for line in f:
            if "LARGE MODELS" in line:
                ring = 1
                continue
            elif "MEDIUM MODELS" in line:
                ring = 2
                continue
            elif "SMALL MODELS" in line:
                ring = 3
                continue

            if ring == 1:
                large_models.append(line.strip())
            elif ring == 2:
                medium_models.append(line.strip())
            elif ring == 3:
                small_models.append(line.strip())
    
    if large:
        for model in large_models:
            pull_ollama_model(model)
    if medium:
        for model in medium_models:
            pull_ollama_model(model)
    if small:
        for model in small_models:
            pull_ollama_model(model)
    
    return large_models, medium_models, small_models
    

def generate_prompt(instruction: str, data: list, task: str) -> str:
    prompt = f"{instruction}:\n\nData:\n{data}\n\nTask:\n{task}.\n\n"
    return prompt


def generate_response(model: str, prompt: str, params: dict) -> Iterator[GenerateResponse]:
    response: Iterator[GenerateResponse] = ollama.generate(
        model=model, 
        prompt=prompt,
        options=params,
        stream=True
    ) 
    return response


def print_response(response: Iterator[GenerateResponse]) -> None:
    try:
        for part in response:
            print(part['response'], end='', flush=True)
    except StopIteration:
        # Response is finished
        print('\n[ERROR]\n')
    print('\n\n[FINISHED]\n')


import os


if __name__ == "__main__":
    # only use cpu for bigger models
    #os.environ["OLLAMA_NO_CUDA"] = "1"

    # generate temperature data
    quantity = 60
    data = generate_temperature_data_file(quantity).values()
    helpers.plot_data(data=data, plot_label="Temperatur", x_label="Zeit (t)", y_label="Temperatur (°C)", title="Temperaturdaten mit stabiler Phase und exponentiellem Anstieg", output_file="temperaturverlauf_exponentiell.png")

    # model
    model = "qwen2.5:14b"
    print(f"\n[MODEL]:\n{model}")

    # model parameter
    params = load_ollama_parameter(isRational=True)
    print(f"\n[PARAMETER]:\n{params}")

    # prompt
    time_diff = 10
    #instruction = f"Hier sind {quantity} Temperaturdaten von einem Prozessor, die im Abstand von {time_diff} Sekunden gemessen wurden. Es sind also Werte über einen Zeitraum von {quantity * time_diff / 60} Minuten. Der erste Messpunkt wurde um 14:00:00 Uhr (HH:MM:SS) gemessen. Interpretiere die Daten in Stichpunkten über einen Zeitraum von je 30 Minuten"
    instruction = f"Analysze the following time series data of temperature measurements taken from a processor"
    data = config.read_data("prompts/error_tsd/test_data.txt")
    tast = f"Identify overall trends"
    prompt = generate_prompt(instruction=instruction, data=data, task=tast)
    print(f"\n[PROMPT]:\n{prompt}")

    # response
    print("\n[RESPONSE]:")
    response = generate_response(model=model, prompt=prompt, params=params)
    print_response(response)