###########################################
##### Functions to use the ollama library
###########################################

import ollama
from typing import Iterator
from ollama import GenerateResponse, ChatResponse

def pull_ollama_model(model: str) -> bool:
    try:
        response = ollama.pull(model=model, stream=True)
        progress_states = set()
        print(f"[PULLING MODEL]: {model}")
        for progress in response:
            if progress.get('status') in progress_states:
                continue
            progress_states.add(progress.get('status'))
            print(f"{progress.get('status')}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    

def generate_response(model: str, prompt: str, params: dict) -> Iterator[GenerateResponse]:
    response: Iterator[GenerateResponse] = ollama.generate(
        model=model, 
        prompt=prompt,
        options=params,
        stream=True,
    ) 
    return response


def print_response(response: Iterator[GenerateResponse]) -> str:
    try:
        response_string = ''
        for part in response:
            print(part['response'], end='', flush=True)
            response_string += part['response']
    except StopIteration:
        # Response is finished
        print('\n[ERROR]\n')
    print('\n\n[FINISHED]\n')
    return response_string


###################################################################################
### TESTING AREA
###################################################################################


def get_current_weather_information(city: str) -> dict:
    return {city: {'temperature': 12, 'humidity': 0.5, 'wind_speed': 10, 'wind_direction': 'NE', 'weather': 'cloudy'}}

def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    print(f"[FUNC] Adding {a} + {b}")
    return int(a) + int(b)

def subtract_two_numbers(a: int, b: int) -> int:
    '''
    Subtracts two numbers

    Args:
    a: int - first number
    b: int - second number

    Returns:
    int - the result of the subtraction
    '''
    print(f"[FUNC] Subtracting {a} - {b}")
    return int(a) - int(b)


def chat(model: str, params: dict) -> str:
    messages = []

    while True:
        print(f"\n[USER]:")
        user_input = input("  > ")
        if user_input == "exit":
            print("Goodbye!")
            break

        response = ollama.chat(
            model=model,
            options=params,
            tools=[add_two_numbers, subtract_two_numbers, get_current_weather_information],
            messages=messages + [{'role': 'user', 'content': user_input}],
            stream=True
        )
        
        print(f"\n[AI]:")
        content = ''
        for part in response:
            content += part['message']['content']
            print(part['message']['content'], end='', flush=True)

        messages += [{'role': 'user', 'content': user_input},
                     {'role': 'assistant', 'content': content}]
        
        print("\n\n[MESSAGES]:", messages)