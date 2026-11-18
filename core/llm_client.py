import os

from openai import OpenAI


def create_client():
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    key = os.getenv("OPENAI_API_KEY")
    if key:
        return OpenAI(api_key=key)

    raise RuntimeError("Set OPENROUTER_API_KEY or OPENAI_API_KEY (or put it in a .env file)")


def default_model():
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1")
    return os.getenv("OPENAI_MODEL", "gpt-4.1")
