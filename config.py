# config.py
# API configuration for baseline scripts.
#
# Do not hard-code real API keys in this file. Set them in your shell instead:
#   export OPENAI_API_KEY=...
#   export DEEPINFRA_API_KEY=...

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "YOUR_DEEPINFRA_API_KEY")

