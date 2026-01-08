
import os
from log.logger import Logger
from src.shapley_values.shapley_agent import ShapleyAgent
from src.zeroshots.utils.utils import pullModel


models = ["artifish/llama3.2-uncensored:latest","llama3.2:latest", "nemotron-3-nano:latest", "gemma3:latest","mistral:7b"]
super_small_path="data/thoughts/supersmall.csv"
cut_data_path="data/thoughts/cut_dataset.csv"
temperatures = [0.0]

def run_shapley():
    logger = Logger(filename="run_log")
    baseURL = os.environ.get("open_ai_host")
    for model in models:
        for temperature in temperatures:
            try:
                pullModel( modelname=model)
                llmModel = ShapleyAgent(
                    model_name=model,
                    dataPath=super_small_path,
                    temperature=temperature,
                    baseURL=baseURL)
                print(f"Predicting for model: {model}, temperature: {temperature}")
                llmModel.predict()
                
            except Exception as e:
                logger.error(f"Model <{model}> failed")
                logger.error(e)