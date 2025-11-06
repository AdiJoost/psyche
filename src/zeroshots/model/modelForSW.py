import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from config.rootPath import getRootPath
from src.zeroshots.utils.utils import loadData


class SWModel():
    baseHeaders = ["TEXT", "label"]

    systemRole = "You are a helpful assistant. You are task with identifying, whether the post came from a person, that is suicidal or not. You get the title of the post and the text from the post. Answere only with 'yes' or 'no' Just do it."


    def __init__(self, model_name: str, dataPath: str, baseURL: str="http://127.0.0.1:11434/v1"):
        self.model_name = model_name
        self.dataPath = dataPath
        self.data = loadData(dataPath, dropLabel=False)
        self.baseURL = baseURL
        self.init_client()

    def init_client(self) -> None:
        self.client = OpenAI(
            base_url=self.baseURL,
            api_key="ollama"
        )
        path = getRootPath().joinpath("data/results/kaggle_data")
        os.makedirs(path, exist_ok=True)
    
    def predict(self) -> None:
        self.data["PREDICTION"] = self.data.progress_apply(lambda x: self.guessIsSuicidal(row=x), axis=1)
        
    def guessIsSuicidal(self, row) -> int:
        messages = self.getMessages(row)
        response = self.client.chat.completions.create(
        model = self.model_name,
        messages=messages
        )
        answere = response.choices[0].message.content
        if answere.strip().lower() == "yes":
            return 1
        if answere.strip().lower() == "no":
            return 0
        return -1

    def getMessages(self, row) -> dict:
        postTitle = row["title"]
        postText = row["usertext"] 
        userQuestion = f"Somewone wrote a post: 'Title:{postTitle}, text:{postText}' Is this person suicidal or not? Answere only with 'yes' or 'no'"
        return [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": userQuestion},
        ]

    def saveResults(self):
        safe_model_name = self.model_name.replace(":", "_").replace("/", "_")

        resultPath = getRootPath().joinpath(f"data/results/kaggle_data/{safe_model_name}.csv")
        self.data.to_csv(resultPath, index=False)