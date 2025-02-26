###########################################
##### Functions to use the ollama library
###########################################

import ollama
from typing import Iterator
from ollama import GenerateResponse

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
        stream=True
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