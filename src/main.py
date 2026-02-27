"""
main.py

Command-line entry point for the MBTI Game of Chicken project.

This module coordinates experiment execution, tournament runs,
and post-experiment analysis based on user-specified arguments.

It loads configuration parameters, validates runtime conditions,
and dispatches execution to run_experiment.py, tournament.py,
or analysis.py as appropriate.
"""
from dotenv import load_dotenv
from tournament import run_experiment, run_both_conditions
from analysis import summarize_results
import os

load_dotenv()

def main():
    print(run_both_conditions())

if __name__ == "__main__":
    main()