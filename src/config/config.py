# Configfile and parameters

# Hier werden Pfade eingestellt und in einer Config File gespeichert
# ... IP der influxDB
# ... Pfad zum LLM

import json
import os

def write_file(type: str, data: any, path: str):
    if type == "list":
        write_file_with_list(data, path)
    elif type == "dict":
        write_file_with_dict(data, path)

def write_file_with_list(data: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for elem in data:
            f.write(f"{elem}\n")

def write_file_with_dict(data: dict, path: str):
    """
    Writes the given dictionary into the config.json, 
    overwrites already existing fields and adds new fields. 
    """

    old_data = read_file()
    if old_data is not None:
        for key in old_data.keys():
            if key not in data.keys():
                data[key] = old_data[key]


    with open(path, "w", encoding="utf-8") as config_file:
        json.dump(data, config_file, ensure_ascii=False, indent=4)


def read_file() -> dict:
    """
    Reads the content of the config.json.

    Return:
    -------
        - the content as a dictionary.
    """

    if not os.path.exists("src/config/config.json"):
        return None
    
    with open("src/config/config.json", "r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    return data


def read_logs_count() -> int:
    """
    Reads the logs count in the config.json and returns the value.

    Return:
    -------
        - the stored value of the logs count
        - -1 if no value stored
    """
    
    data = read_file()
    if data.get("logs") is not None:
        return data.get("logs")
    else:
        return -1
    

def get_content(setup_usr: str, logs: bool, ollama_param: bool) -> dict:
    """
    Returns the specifically requested content from the config.json.

    Param:
    ------
    Every Parameter is optional!
        - setup_usr (str): A string with the name of the user ("noahthebuzz", "dbisai", ...)
        - logs (bool): A flag used to get logs counter 
        - ollama_param (str): The setup for the LLM ("rational", "creative", "default")

        
    Return:
    -------
    Returns the requested data!
        - setup_usr: -> {setup_usr: dict{"Processor", "RAM", "GPU", "OS"}}
        - logs: -> {"logs": int}
        - ollama_param: -> {"param": dict{"mirostat", "mirostat_eta", "mirostat_tau", "num_ctx", "repeat_last_n", "repeat_penalty", "temperature", "seed", "num_predict", "top_k", "top_p", "min_p"}}
    """

    data = read_file()
    new_data = {}
    if setup_usr:
        new_data.update({setup_usr: data.get("setup").get(setup_usr)})
    if logs:
        if data.get("logs") is not None:
            new_data.update({"logs": data.get("logs")})
        else:
            new_data.update({"logs": 0})
    if ollama_param is not None:
        if ollama_param :
            new_data.update({"param": data.get("ollama_parameter").get("rational")})
        else:
            new_data.update({"param": data.get("ollama_parameter").get("creative")})    

    return new_data



'''
if __name__ == "__main__":
    #data = {"setup": {"setup_noahthebuzz": {"Processor": "Intel Core i7-12700H", "RAM": "16 GB DDR4", "GPU": "NVIDIA GeForce RTX 4060 (8GB GDDR6X, 3072 CUDA Cores)", "OS": "Ubuntu 22.04"}, "setup_dbisai": {"Processor": "Intel Core i9-14900K", "RAM": "64 GB DDR5", "GPU": "NVIDIA GeForce RTX 4090 (24GB GDDR6X, 16384 CUDA Cores)", "OS": "Ubuntu 22.04"}}}
    #data = {"logs": 0}
    
    write_file(data)
'''

'''
if __name__ == "__main__":
    print(get_content(setup_usr="noahthebuzz", logs=True, ollama_param=1))
'''