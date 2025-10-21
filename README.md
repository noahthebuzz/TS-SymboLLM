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

'''
    $ python3 main.py
'''

## LLMs (anzeigen, herunterladen, löschen, ...)
-> vorhandene Modelle anzeigen
'''
$ funcllama --list models
'''

-> via ollama herunterladen
'''
$ funcllama --pull %_ollama.model.name_%
'''

-> vorhandenes Modell löschen
'''
$ funcllama --delete %_ollama.model.name_%
'''

## Datensätze 
-> vorhandene Datensätze anzeigen
'''
$ funcllama --list datasets
'''

-> Datensätze neu laden
'''
$ funcllama --reload datasets
'''

## Promptfiles
-> vorhandene Promptfiles anzeigen
'''
$ funcllama --list prompts
'''

-> Prompts neu laden
'''
$ funcllama --reload prompts
'''

## Parameter
-> vorhandene Parameterfiles anzeigen
'''
$ funcllama --list params
'''

-> Parameterfiles neu laden
'''
$ funcllama --reload params
'''

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
'''
$ mkdir %project/folder%
'''

### change directory to project folder
'''
$ cd %project/folder%
'''

### clone the repository
'''
$ git clone %ssh_link%
'''

### change to repository directory
'''
$ cd %reponame%
'''

### create virtual environment (venv)
(.venv can be replaced by whatever name; maybe you have to use python3 instead of python)
'''
$ python -m venv .venv
'''

### activate virtual environment
'''
$ source .venv/bin/activate
'''

### install requirements.txt
'''
$ pip install -r requirements.txt
'''

### If everything went as expected, you should be able to start the programm now with:
'''
$ python3 main.py
'''
