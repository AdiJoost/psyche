from src.zeroshots.run_zero_shot import run, run_suicide_watch
from dotenv import load_dotenv

def main():
    print("Application started")
    load_dotenv()
    run_suicide_watch()

if __name__ == "__main__":
    main()