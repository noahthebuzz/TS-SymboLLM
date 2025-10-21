[Installation Guide](#installation-guide) [Usage](#usage)

# Ordner-Struktur
LLM_ERROR/
    .venv/
    logs/
    prompts/
    datasets/
    params/
    src/
    app.py
    LICENSE
    .gitignore
    .gitattributes
    
    
# Usage
## Start the LLM Error Handler
```bash
$ python3 main.py
```

## LLMs (anzeigen, herunterladen, löschen, ...)
-> vorhandene Modelle anzeigen
```bash
$ funcllama --list models
```

-> via ollama herunterladen
```bash
$ funcllama --pull %_ollama.model.name_%
```

-> vorhandenes Modell löschen
```bash
$ funcllama --delete %_ollama.model.name_%
```

## Datensätze 
-> vorhandene Datensätze anzeigen
```bash
$ funcllama --list datasets
```

-> Datensätze neu laden
```bash
$ funcllama --reload datasets
```

## Promptfiles
-> vorhandene Promptfiles anzeigen
```bash
$ funcllama --list prompts
```

-> Prompts neu laden
```bash
$ funcllama --reload prompts
```

## Parameter
-> vorhandene Parameterfiles anzeigen
```bash
$ funcllama --list params
```

-> Parameterfiles neu laden
```bash
$ funcllama --reload params
```

## Run (bisher nur Generate; kein Chat)
-> List models
-> Enter model to use
-> List paramfiles
-> Enter params to use
-> List promptfiles
-> Enter promptfile to use
-> Enter name of logfile (can include folder %folder/name%; default: folder -> current date, name -> current time)
-> Output generation


# Installation Guide
### create project folder
```bash
$ mkdir %project/folder%
```

### change directory to project folder
```bash
$ cd %project/folder%
```

### clone the repository
```bash
$ git clone %ssh_link%
```

### change to repository directory
```bash
$ cd %reponame%
```

### create virtual environment (venv)
(.venv can be replaced by whatever name; maybe you have to use python3 instead of python)
```bash
$ python -m venv .venv
```

### activate virtual environment
```bash
$ source .venv/bin/activate
```

### install requirements.txt
```bash
$ pip install -r requirements.txt
```

### If everything went as expected, you should be able to start the programm now with:
```bash
$ python3 main.py
```
