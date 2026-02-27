"""
list_models.py

Utility script for listing available language models and adapter variants.

This module verifies that required base models and fine-tuned adapters
(e.g., via Ollama or other backends) are accessible prior to running experiments.

It serves as an environment validation tool and does not contain
game logic or experiment orchestration.
"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    # Print model name and supported methods
    print(m.name, getattr(m, "supported_actions", None))
