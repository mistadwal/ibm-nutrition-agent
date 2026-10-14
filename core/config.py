"""
core/config.py
Centralised configuration loaded from .env / environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from the project root (one level above this file)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")


class Settings(BaseModel):
    # IBM watsonx.ai
    watsonx_api_key: str = Field(default_factory=lambda: os.environ.get("WATSONX_API_KEY", ""))
    watsonx_project_id: str = Field(default_factory=lambda: os.environ.get("WATSONX_PROJECT_ID", ""))
    watsonx_url: str = Field(default_factory=lambda: os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"))

    # Model IDs
    granite_llm_model: str = Field(default_factory=lambda: os.environ.get("GRANITE_LLM_MODEL", "ibm/granite-13b-instruct-v2"))
    granite_embedding_model: str = Field(default_factory=lambda: os.environ.get("GRANITE_EMBEDDING_MODEL", "ibm/slate-125m-english-rtrvr"))

    # Vector store
    chroma_persist_dir: str = Field(default_factory=lambda: os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db"))

    # USDA food API
    usda_api_key: str = Field(default_factory=lambda: os.environ.get("USDA_API_KEY", "DEMO_KEY"))

    # App
    app_host: str = Field(default_factory=lambda: os.environ.get("APP_HOST", "0.0.0.0"))
    app_port: int = Field(default_factory=lambda: int(os.environ.get("APP_PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.environ.get("LOG_LEVEL", "info"))

    class Config:
        arbitrary_types_allowed = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
