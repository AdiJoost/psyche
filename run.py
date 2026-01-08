from src.shapley_values.run_shapley import run_shapley
from src.zeroshots.run_zero_shot import run, run_suicide_watch
from src.agentic_shot.agentic_shot import run_agent
import asyncio


def main():
    run_shapley()

if __name__ == "__main__":
    #asyncio.run(run_agent())
    main()