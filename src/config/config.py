# Configfile and parameters

# Hier werden Pfade eingestellt und in einer Config File gespeichert
# ... IP der influxDB
# ... Pfad zum LLM

import json
import os



def write_file(data: dict):  
    old_data = read_file()
    if old_data is not None:
        for key in old_data.keys():
            if key not in data.keys():
                data[key] = old_data[key]

    #print(f"TEST: {data}")

    with open("src/config/config.json", "w", encoding="utf-8") as config_file:
        json.dump(data, config_file, ensure_ascii=False, indent=4)



def read_file():
    if not os.path.exists("src/config/config.json"):
        return None
    
    with open("src/config/config.json", "r", encoding="utf-8") as config_file:
        data = json.load(config_file)
    return data

"""
if __name__ == "__main__":
    #data = {"setup": {"setup_noahthebuzz": {"Processor": "Intel Core i7-12700H", "RAM": "16 GB DDR4", "GPU": "NVIDIA GeForce RTX 4060 (8GB GDDR6X, 3072 CUDA Cores)", "OS": "Ubuntu 22.04"}, "setup_dbisai": {"Processor": "Intel Core i9-14900K", "RAM": "64 GB DDR5", "GPU": "NVIDIA GeForce RTX 4090 (24GB GDDR6X, 16384 CUDA Cores)", "OS": "Ubuntu 22.04"}}}
    #data = {"logs": 0}
    
    write_file(data)
"""