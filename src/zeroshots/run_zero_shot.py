import os
from log.logger import Logger
from src.zeroshots.model.model import LLMModel
from src.zeroshots.utils.utils import pullModel


models = [
    "llama3.2:latest",
    "gpt-oss:latest",
    "deepseek-r1:latest",
    "gpt-oss:120b",
    "gemma3:27b",
    "mistral:7b"]

dataPathDepression = "data/thoughts/Depression.csv"
dataPathNeutral = "data/thoughts/Neutral.csv"
dataPathSuicidal = "data/thoughts/Suicadal_tendencies_data.csv"

def run():
    logger = Logger(filename="run_log")
    baseURL = os.environ.get("open_ai_host")
    for model in models:
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