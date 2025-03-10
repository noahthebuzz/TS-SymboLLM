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
    """
    Get the current weather information for a city

    Args:
        city (str): The city to get the weather information for

    Returns:
        dict: The weather information containing temperature, humidity, wind speed, wind direction, and weather condition
    """
    return {city: {'temperature': 12, 'humidity': 0.5, 'wind_speed': 10, 'wind_direction': 'NE', 'weather condition': 'cloudy'}}

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
    """
    Subtract two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The subtraction of the two numbers
    """
    print(f"[FUNC] Subtracting {a} - {b}")
    return int(a) - int(b)


def chat(model: str, params: dict) -> str:
    messages = []

    available_functions = {
        'add_two_numbers': add_two_numbers,
        'subtract_two_numbers': subtract_two_numbers,
        'get_current_weather_information': get_current_weather_information,
    }

    while True:
        print(f"\n[USER]:")
        user_input = input("  > ")
        if user_input == "exit":
            print("Goodbye!")
            break

        messages.append({'role': 'user', 'content': user_input})

        response: ChatResponse = ollama.chat(
            model=model,
            options=params,
            messages=messages,
            tools=[add_two_numbers, subtract_two_numbers, get_current_weather_information],
        )

        if response.message.tool_calls:
        # There may be multiple tool calls in the response
            for tool in response.message.tool_calls:
                # Ensure the function is available, and then call it
                if function_to_call := available_functions.get(tool.function.name):
                    print('[INFO] Calling function:', tool.function.name)
                    print('[INFO] Arguments:', tool.function.arguments)
                    output = function_to_call(**tool.function.arguments)
                    print('[INFO] Function output:', output)
                else:
                    print('[INFO] Function', tool.function.name, 'not found')

        print(f"[DEBUG]: {response.message.content}")
        print(f"[MESSAGES]: {messages}")

        # Only needed to chat with the model using the tool call results
        if response.message.tool_calls:
            # Add the function response to messages for the model to use
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})

            # Get final response from model with function outputs
            final_response = ollama.chat('qwen2.5:3b', messages=messages, options=params)
            print('\n[AI]:', final_response.message.content)

        else:
            print('No tool calls returned from model')