import itertools
import os
from typing import List
import numpy as np
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import math
import json

from config.rootPath import getRootPath
from src.zeroshots.utils.utils import loadData


class ShapleyAgent():
    baseHeaders = ["TEXT", "label"]

    systemRole = "You are a helpful assistant. You are task with identifying, whether the post came from a person, that is suicidal or not. You get the title of the post and the text from the post. Answere only with 'yes' or 'no' Just do it."


    def __init__(self, model_name: str, dataPath: str, baseURL: str="http://127.0.0.1:11434/v1", temperature=0.0):
        self.model_name = model_name
        self.dataPath = dataPath
        self.temperature = temperature
        
        self.data = loadData(dataPath, dropLabel=False)
        self.baseURL = baseURL

        self.save_name = self.getSaveName()
        self.path = getRootPath().joinpath(f"data/results/shapley/{self.save_name}")
        os.makedirs(self.path, exist_ok=True)
        self.init_client()

    def init_client(self) -> None:
        self.client = OpenAI(
            base_url=self.baseURL,
            api_key="ollama"
        )
        
    
    def predict(self) -> None:
        for row in self.data.itertuples():
            words = row.usertext.split()
            values_df = self.get_values_df(words)
            shapley_values = self.calculate_shapley_values(words=words, df=values_df)
            self.save_shapley_values(text=row.usertext, shapley_values=shapley_values, number=row.Index)

    def save_shapley_values(self, text, shapley_values, number) -> None:
        json_data = {
            "text": text,
            "shapley_values": shapley_values,
            "total_payoff": sum(shapley_values)
        }
        with open(self.path.joinpath(f"shapley_{number}.json"), "w") as f:
            json.dump(json_data, f, indent=4)

    def get_values_df(self, words) -> pd.DataFrame:
        mask = self._getWordMask(words)
        rows = []
        for bits in mask:
            mask_str = ''.join(str(b) for b in bits)
            assembled = ' '.join([w for w, b in zip(words, bits) if b == 1])
            value = self.evaluate_fn(assembled)
            rows.append({'mask': mask_str, 'text': assembled, 'value': value})

        return pd.DataFrame(rows)

    def calculate_shapley_values(self, words, df) -> List:
        df = df.set_index("mask")
        values = []
        n = len(words)
        n_factorial = math.factorial(n)
        for i in range(len(words)):
            phy = 0.0
            for row in df.itertuples():
                if row.Index[i] == "1":
                    continue
                val_s = float(row.value)
                new_mask = self.change_bit(row.Index, i, "1")
                val_s_i = float(df.loc[new_mask, "value"])
                players = row.Index.count("1")
                denominator = float(math.factorial(players) * math.factorial((n - players - 1))) / float(n_factorial)
                phy += denominator * (val_s_i - val_s)
            values.append(phy)
        return values

    def evaluate_fn(self, text):
        messages = self.getMessages(text)
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

    def _getWordMask(self, words) -> list:
        n = len(words)
        masks = []
        for bits in itertools.product([0, 1], repeat=n):
            masks.append(list(bits))
        return masks

    def getMessages(self, text) -> dict:
        userQuestion = f"Somewone wrote a post: {text}' Is this person suicidal or not? Answere only with 'yes' or 'no'"
        return [
            {"role": "system", "content": self.systemRole},
            {"role": "user", "content": userQuestion},
        ]
    
    def change_bit(self, s, i, new_bit):
        return s[:i] + new_bit + s[i+1:]
    
    def getSaveName(self):
        return self.model_name.replace(":", "_").replace("/", "_")