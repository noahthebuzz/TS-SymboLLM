from ollama import chat, ChatResponse, list, GenerateResponse, generate

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

def load_ollama_parameter(isRational: bool):
    options = config.get_content(ollama_param=True)

def generate_temperature_data_file():
    datalist = helpers.generate_data("temperature").tolist()
    config.write_file("list", datalist, "prompts/error_tsd/test_data.txt")


if __name__ == "__main__":
    generate_temperature_data_file()