"""
Central configuration for the AI HR Recruitment Assistant.
Loads settings from environment variables (.env file) so no secrets
ever live in source code, and no OpenRouter model name is hard-coded
anywhere else in the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "AI HR Recruitment Assistant")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/knowledge_base/chroma_store")

UPLOAD_DIR = "uploads"
REPORTS_DIR = "reports"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"


class ConfigError(Exception):
    """Raised when required configuration (like an API key) is missing."""


def require_api_key() -> None:
    """Raise a clear, friendly error if no OpenRouter API key has been configured."""
    if not OPENROUTER_API_KEY or not OPENROUTER_API_KEY.strip():
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add a free "
            "API key from https://openrouter.ai/keys"
        )
