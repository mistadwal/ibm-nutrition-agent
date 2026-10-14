"""
core/vector_store.py
ChromaDB-backed vector store for RAG across nutrition knowledge documents.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from core.config import get_settings
from core.llm import get_embeddings

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "nutrition_knowledge"


def get_vector_store() -> Chroma:
    """Return (or create) the persistent ChromaDB vector store."""
    settings = get_settings()
    persist_dir = str(Path(settings.chroma_persist_dir).resolve())
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=_COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir,
    )
    logger.info("ChromaDB vector store ready at: %s", persist_dir)
    return store


def ingest_documents(docs: list[Document], batch_size: int = 100) -> int:
    """
    Add documents to the vector store in batches.
    Returns the total count of documents ingested.
    """
    store = get_vector_store()
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        store.add_documents(batch)
        total += len(batch)
        logger.debug("Ingested batch %d/%d (%d docs)", i // batch_size + 1, -(-len(docs) // batch_size), len(batch))
    logger.info("Total documents ingested: %d", total)
    return total


def similarity_search(query: str, k: int = 5, filter: Optional[dict] = None) -> list[Document]:
    """Run a similarity search against the nutrition knowledge base."""
    store = get_vector_store()
    if filter:
        return store.similarity_search(query, k=k, filter=filter)
    return store.similarity_search(query, k=k)
