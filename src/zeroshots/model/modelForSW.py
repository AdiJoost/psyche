import itertools
import os
import numpy as np
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import math

from config.rootPath import getRootPath
from src.zeroshots.utils.utils import loadData


class SWModel():
    baseHeaders = ["TEXT", "label"]

    systemRole = "You are a helpful assistant. You are task with identifying, whether the post came from a person, that is suicidal or not. You get the title of the post and the text from the post. Answere only with 'yes' or 'no' Just do it."


    def __init__(self, model_name: str, dataPath: str, modelNamePrefix: str, baseURL: str="http://127.0.0.1:11434/v1", temperature=0.0, maxWords: int=10):
        self.model_name = model_name
        self.dataPath = dataPath
        self.modelNamePrefix = modelNamePrefix
        self.temperature = temperature
        self.maxWords = maxWords
        self.data = loadData(dataPath, dropLabel=False)
        self.baseURL = baseURL
        self.init_client()

    def init_client(self) -> None:
        self.client = OpenAI(
            base_url=self.baseURL,
            api_key="ollama"
        )
        saveName = self.getSaveName()
        path = getRootPath().joinpath(f"data/results/{saveName}/{self.modelNamePrefix}")
        os.makedirs(path, exist_ok=True)
    
    def predict(self) -> None:
        self.data["PREDICTION"] = self.data.progress_apply(lambda x: self.guessIsSuicidal(row=x), axis=1)

    def predict_shaply(self) -> None:
        shapley_values = self.data.progress_apply(lambda x: self.calculateShapley(row=x), axis=1)
        shapley_df = pd.DataFrame(shapley_values.tolist(), columns=list(range(self.maxWords)))
        self.data = pd.concat([self.data, shapley_df], axis=1)

    def create_dataset(self) -> None:
        pass

    def calculateShapley(self, row) -> np.array:
        words = row["usertext"]
        if len(words) > self.maxWords:
            print(f"Truncating words from {len(words)} to {self.maxWords}")
            words = words[:self.maxWords]
        mask = self._getWordMask(words)
        
        print(f"Predicting for: {words}")
        for bits in tqdm(mask):
            subset = self._getString(words, bits)
            bits.append(self.guessIsSuicidalFromMessage(subset))
        values = np.zeros(self.maxWords)

        n = len(words)
        n_factorial = math.factorial(n)
        for i in range(len(words)):
            phy = 0
            for m in mask:
                if m[i] == 1:
                    continue
                s = m[:-1]
                val_s = m[-1]
                lowMask = m[:-1].copy()
                lowMask[i] = 1
                matching = [coal for coal in mask if coal[:-1] == lowMask]
                if matching:
                    val_s_i = matching[0][-1]
                else:
                    print("Value not found")
                    val_s_i = 0
                number_of_players = sum(s)
                denominator = float(math.factorial(number_of_players) * math.factorial((n - number_of_players - 1))) / float(n_factorial)
                #printIt(val_s_i, val_s, denominator, number_of_players, phy, add)
                phy += denominator * (val_s_i - val_s)
            values[i] = phy
        print(f"Shapley values for {words   }: {values}")
        return values

    def _getWordMask(self, words) -> list:
        n = len(words)
        masks = []
        for bits in itertools.product([0, 1], repeat=n):
            masks.append(list(bits))
        return masks

    def _getString(self,words, bits):
        return [w for w, b in zip(words, bits) if b == 1]

    def guessIsSuicidal(self, row) -> int:
        messages = self.getMessages(row)
        response = self.client.chat.completions.create(
        model = self.model_name,
        messages=messages,
        temperature=self.temperature
        )
        answere = response.choices[0].message.content
        if answere.strip().lower() == "yes":
            return 1
        if answere.strip().lower() == "no":
            return 0
        return -1
    
    def guessIsSuicidalFromMessage(self, text) -> int:
        messages = self.getMessageFromText(text)
        response = self.client.chat.completions.create(
        model = self.model_name,
        messages=messages,
        temperature=self.temperature
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
    
    def getMessageFromText(self, text) -> dict:
        userQuestion = f"Somewone wrote a post: 'Title:, text:{text}' Is this person suicidal or not? Answere only with 'yes' or 'no'"
        return [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": userQuestion},
        ]
    def getSaveName(self):
        return self.model_name.replace(":", "_").replace("/", "_")

    def saveResults(self):
        safe_model_name = self.getSaveName()
        resultPath = getRootPath().joinpath(f"data/results/{safe_model_name}/{self.modelNamePrefix}/temp_{self.temperature}.csv")
        self.data.to_csv(resultPath, index=False)