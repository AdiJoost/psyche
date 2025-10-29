import os
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from config.rootPath import getRootPath
from src.zeroshots.utils.utils import loadData


class LLMModel():
    baseHeaders = ["TEXT", "label"]

    systemRole = "You are a helpful assistant. You are task with identifying, whether the text came from a person, that is suicidal or not. Answere only with 'yes' or 'no'"


    def __init__(self, model_name: str, neutralPath: str, suicidePath: str, depressionPath: str, baseURL: str="http://127.0.0.1:11434/v1"):
        self.model_name = model_name
        self.neutralPath = neutralPath
        self.suicidePath = suicidePath
        self.depressionPath = depressionPath
        self.neutralData = loadData(neutralPath)
        self.suicideData = loadData(suicidePath)
        self.depressionData = loadData(depressionPath)
        self.baseURL = baseURL
        self.init_client()

    def init_client(self) -> None:
        self.client = OpenAI(
            base_url=self.baseURL,
            api_key="ollama"
        )
        path = getRootPath().joinpath("data/results")
        os.makedirs(path, exist_ok=True)
    
    def predict(self) -> None:
        if(self.neutralData):
            self.neutralData["PREDICTION"] = self.neutralData["TEXT"].progress_apply(lambda x: self.guessIsSuicidal(message=x))
        if(self.suicideData):
            self.suicideData["PREDICTION"] = self.suicideData["TEXT"].progress_apply(lambda x: self.guessIsSuicidal(message=x))
        if (self.depressionData):
            self.depressionData["PREDICTION"] = self.depressionData["TEXT"].progress_apply(lambda x: self.guessIsSuicidal(message=x))
        
    def guessIsSuicidal(self, message: dict) -> int:
        messages = self.getMessages(message)
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

    def getMessages(self, userMessage: str) -> dict:
        userQuestion = f"Somewone wrote: '{userMessage}' Is this person suicidal or not? Answere only with 'yes' or 'no'"
        return [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": userQuestion},
        ]

    def saveResults(self):
        safe_model_name = self.model_name.replace(":", "_")

        pathNeutral = getRootPath().joinpath(f"data/results/{safe_model_name}_neutral.csv")
        self.neutralData.to_csv(pathNeutral, index=False)

        pathSuicidal = getRootPath().joinpath(f"data/results/{safe_model_name}_suicidal.csv")
        self.suicideData.to_csv(pathSuicidal, index=False)

        pathDepression = getRootPath().joinpath(f"data/results/{safe_model_name}_depression.csv")
        self.depressionData.to_csv(pathDepression, index=False)