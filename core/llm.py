"""
core/llm.py
Builds the IBM watsonx.ai Granite LLM and embedding objects.
Falls back gracefully to a local stub when credentials are absent
(useful for unit tests / local demos without IBM Cloud access).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_llm(temperature: float = 0.3, max_new_tokens: int = 1024):
    """Return a cached IBM Granite LLM instance."""
    settings = get_settings()

    if not settings.watsonx_api_key or settings.watsonx_api_key == "your_ibm_cloud_api_key_here":
        logger.warning("No IBM watsonx credentials found — using FakeLLM stub.")
        return _fake_llm()

    try:
        from langchain_ibm import WatsonxLLM

        params: dict[str, Any] = {
            "decoding_method": "greedy",
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 1,
            "temperature": temperature,
            "repetition_penalty": 1.1,
        }
        llm = WatsonxLLM(
            model_id=settings.granite_llm_model,
            url=settings.watsonx_url,
            apikey=settings.watsonx_api_key,
            project_id=settings.watsonx_project_id,
            params=params,
        )
        logger.info("Granite LLM initialised: %s", settings.granite_llm_model)
        return llm
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialise Granite LLM: %s — falling back to stub.", exc)
        return _fake_llm()


def _fake_llm():
    """A minimal LLM stub for offline/demo mode."""
    from langchain_core.language_models.llms import LLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun

    class _FakeLLM(LLM):
        @property
        def _llm_type(self) -> str:
            return "fake"

        def _call(
            self,
            prompt: str,
            stop=None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs,
        ) -> str:
            return (
                "[DEMO MODE — connect IBM watsonx.ai for real Granite responses]\n"
                f"Received prompt ({len(prompt)} chars). Please configure .env with valid credentials."
            )

    return _FakeLLM()


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_embeddings():
    """Return embedding model (Granite via watsonx or local sentence-transformers fallback)."""
    settings = get_settings()

    if not settings.watsonx_api_key or settings.watsonx_api_key == "your_ibm_cloud_api_key_here":
        logger.warning("No IBM watsonx credentials — using local sentence-transformers embeddings.")
        return _local_embeddings()

    try:
        from langchain_ibm import WatsonxEmbeddings
        from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

        embeddings = WatsonxEmbeddings(
            model_id=settings.granite_embedding_model,
            url=settings.watsonx_url,
            apikey=settings.watsonx_api_key,
            project_id=settings.watsonx_project_id,
        )
        logger.info("Granite Embeddings initialised: %s", settings.granite_embedding_model)
        return embeddings
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialise Granite Embeddings: %s — falling back to local.", exc)
        return _local_embeddings()


def _local_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
