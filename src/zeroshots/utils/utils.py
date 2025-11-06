
import os
import pandas as pd
from config.rootPath import getRootPath
from ollama import Client


def loadData(dataPath: str, dropLabel=True) -> pd.DataFrame:
    path = getRootPath().joinpath(dataPath)
    df = pd.read_csv(path)
    if dropLabel:
        df.drop(columns=["Label"], inplace=True)
    return df

def pullModel(modelname:str):
    print(f"Pulling model: {modelname}")
    host = os.environ.get("ollama_host")
    client = Client(host=host)
    client.pull(modelname)