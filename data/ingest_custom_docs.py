"""
data/ingest_custom_docs.py
─────────────────────────
Utility script to ingest custom documents (text files, PDFs, or URLs)
into the ChromaDB nutrition knowledge base.

Usage:
    python data/ingest_custom_docs.py --path /path/to/docs/
    python data/ingest_custom_docs.py --url https://example.com/nutrition-guide.html
"""
from __future__ import annotations

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_text_files(directory: str) -> list[Document]:
    """Load all .txt files from a directory."""
    docs = []
    for path in Path(directory).rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": str(path), "title": path.stem}))
    logger.info("Loaded %d text files from %s", len(docs), directory)
    return docs


def load_url(url: str) -> list[Document]:
    """Load and parse a web page."""
    try:
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        docs = loader.load()
        for doc in docs:
            doc.metadata.setdefault("source", url)
        logger.info("Loaded %d document(s) from URL: %s", len(docs), url)
        return docs
    except Exception as exc:
        logger.error("Failed to load URL %s: %s", url, exc)
        return []


def split_documents(docs: list[Document], chunk_size: int = 800, overlap: int = 100) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_docs = splitter.split_documents(docs)
    logger.info("Split into %d chunks (chunk_size=%d)", len(split_docs), chunk_size)
    return split_docs


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Nutrition Knowledge Base")
    parser.add_argument("--path", help="Directory of text files to ingest")
    parser.add_argument("--url", help="URL to scrape and ingest")
    parser.add_argument("--chunk-size", type=int, default=800, help="Text chunk size (default 800)")
    args = parser.parse_args()

    if not args.path and not args.url:
        parser.print_help()
        sys.exit(1)

    from core.vector_store import ingest_documents

    raw_docs: list[Document] = []
    if args.path:
        raw_docs.extend(load_text_files(args.path))
    if args.url:
        raw_docs.extend(load_url(args.url))

    if not raw_docs:
        logger.error("No documents loaded. Check your --path or --url argument.")
        sys.exit(1)

    chunks = split_documents(raw_docs, chunk_size=args.chunk_size)
    count = ingest_documents(chunks)
    logger.info("✅ Successfully ingested %d chunks into ChromaDB.", count)


if __name__ == "__main__":
    main()
