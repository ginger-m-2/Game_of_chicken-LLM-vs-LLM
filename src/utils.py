"""
utils.py

Contains shared utility functions used across the project.

This module provides reusable helpers such as:
    - Deterministic seed generation
    - Structured logging helpers
    - JSONL reading and writing
    - Small formatting and path utilities

It contains no experiment logic or model behavior.
"""
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)
