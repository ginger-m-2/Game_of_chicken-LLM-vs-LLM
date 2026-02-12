from dotenv import load_dotenv
from tournament import run_tournament
from analysis import summarize_results
import os

load_dotenv()

def main():
    print(run_tournament())
    summarize_results()

if __name__ == "__main__":
    main()