import os

FEATURE_STORAGE_MODE = "local"
CONFIG_FILE = "secret.yaml"
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_CONFIG = {"timeout": 60, "max_retries": 1, "api_key": os.environ.get("OPENAI_API_KEY")}
WELCOME_MESSAGE = """
Hi <b style='color: #f5145f'>{name}</b>, welcome to the Food Advisor Bot! 🤖
Here, you can find the best restaurants in Vietnam 🇻🇳.
Discover the best dining spots in <b>Ho Chi Minh</b> and <b>Hanoi</b>! 🍽️
🍲🍹 Bon appétit and happy exploring! 🥳🎉
> <b>Note:</b> Tune your preferences to get the best recommendations fit your needs.  
"""
