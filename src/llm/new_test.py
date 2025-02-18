import ollama

response = ollama.generate(
        model='qwen2.5:7b', 
        prompt='Hier sind 10 Temperaturdaten von einem Prozessor, die immer im Abstand von 10 Sekunden gemessen wurden. Gib mir eine Interpretation der Daten: 50 45 43 46 52 55 50 60 65 54',
        options={
            'mirostat': 2,
            'seed': 0,
            'temperature': 0,
            'top_k': 0,
            'top_p': 0.1,
            'min_p': 0.0,
            'num_ctx': 2048,
            'num_gpu': 1
        },
        stream=True
    )

try:
    for part in response:
        print(part['response'], end='', flush=True)
    print('\n')
except StopIteration:
    None