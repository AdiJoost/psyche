import os
from src.zeroshots.model.model import LLMModel
from src.zeroshots.utils.utils import pullModel


models = [
    "llama3.2:latest",
    "gpt-oss:latest",
    "deepseek-r1:latest"]

dataPathDepression = "data/thoughts/Depression.csv"
dataPathNeutral = "data/thoughts/Neutralsmall.csv"
dataPathSuicidal = "data/thoughts/Suicadal_tendencies_data.csv"

def run():
    baseURL = os.environ.get("open_ai_host")
    for model in models:
        pullModel( modelname=model)
        llmModel = LLMModel(
            model_name=model,
            neutralPath=dataPathNeutral,
            suicidePath=dataPathSuicidal,
            depressionPath=dataPathDepression,
            baseURL=baseURL)
        print(f"Predicting for model: {model}")
        llmModel.predict()
        print(f"Saving results for model: {model}")
        llmModel.saveResults()