import os
from log.logger import Logger
from src.zeroshots.model.model import LLMModel
from src.zeroshots.model.modelForSW import SWModel
from src.zeroshots.utils.utils import pullModel

"""
models = [
    "artifish/llama3.2-uncensored:latest",
    "llama3.2:latest",
    "gpt-oss:latest",
    "deepseek-r1:latest",
    "gpt-oss:120b",
    "gemma3:27b",
    "mistral:7b"]"""

models = ["artifish/llama3.2-uncensored:latest",
          "llama3.2:latest"]

temperatures = [0.0, 0.2, 0.5, 1.0, 2.0]

dataPathDepression = "data/thoughts/Depression.csv"
dataPathNeutral = "data/thoughts/Neutral.csv"
dataPathSuicidal = "data/thoughts/Suicadal_tendencies_data.csv"
suicide_watch_path = "data/thoughts/sample_data_1.csv"
kaggle_path="data/thoughts/kaggle_data.csv"

def run():
    logger = Logger(filename="run_log")
    baseURL = os.environ.get("open_ai_host")
    for model in models:
        for temperature in temperatures:
            try:
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
            except Exception as e:
                logger.error(f"Model <{model}> failed")
                logger.error(e)

def run_suicide_watch():
    logger = Logger(filename="run_log")
    baseURL = os.environ.get("open_ai_host")
    for model in models:
        for temperature in temperatures:
            try:
                pullModel( modelname=model)
                llmModel = SWModel(
                    model_name=model,
                    dataPath=kaggle_path,
                    modelNamePrefix="zeroShot",
                    temperature=temperature,
                    baseURL=baseURL)
                print(f"Predicting for model: {model}")
                llmModel.predict()
                print(f"Saving results for model: {model}")
                llmModel.saveResults()
            except Exception as e:
                logger.error(f"Model <{model}> failed")
                logger.error(e)