"""
Run this ONCE to chunk, embed, and index all documents into ChromaDB.
Usage (from project root): python scripts/build_kb.py
"""
import sys
import json
from pathlib import Path

# Ensure project root is in path regardless of where script is called from
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root so relative paths work
import os
os.chdir(PROJECT_ROOT)

from rag.loader import DocumentLoader
from rag.chunker import SmartChunker
from rag.embedder import EmbeddingEngine
from rag.vectorstore import PassportVectorStore
from loguru import logger


def main():
    logger.info("=== Building Passport Knowledge Base ===")

    loader = DocumentLoader("data/raw", "passport_docs")
    chunker = SmartChunker()
    embedder = EmbeddingEngine()

    logger.info("Step 1: Loading documents...")
    docs = loader.load_all()
    if not docs:
        logger.error("No documents found in data/raw/. Aborting.")
        return
    logger.info(f"Loaded {len(docs)} documents")

    logger.info("Step 2: Chunking...")
    chunks = chunker.chunk_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info("Chunks saved to data/processed/chunks.jsonl")

    logger.info("Step 3: Indexing into ChromaDB...")
    vectorstore = PassportVectorStore(embedder)
    vectorstore.add_chunks(chunks)

    logger.success(f"Done! Stats: {vectorstore.get_collection_stats()}")


if __name__ == "__main__":
    main()
