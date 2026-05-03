"""
Run full evaluation suite.
Usage: python scripts/run_eval.py
"""
import sys
sys.path.insert(0, ".")

from rag.embedder import EmbeddingEngine
from rag.vectorstore import PassportVectorStore
from rag.retriever import HybridRetriever
from rag.loader import DocumentLoader
from rag.chunker import SmartChunker
from rag.generator import ResponseGenerator
from evaluation.evaluator import PassportRAGEvaluator
from pathlib import Path


def main():
    embedder = EmbeddingEngine()
    loader = DocumentLoader("data/raw")
    chunker = SmartChunker()
    docs = loader.load_all()
    chunks = chunker.chunk_documents(docs)
    vectorstore = PassportVectorStore(embedder)
    retriever = HybridRetriever(vectorstore, chunks)
    generator = ResponseGenerator(language="en")
    evaluator = PassportRAGEvaluator(retriever, generator)

    test_set = evaluator.load_test_set("data/evaluation/test_questions_en.json")
    df = evaluator.evaluate(test_set)

    Path("evaluation").mkdir(parents=True, exist_ok=True)
    df.to_csv("evaluation/results.csv", index=False)
    print("Results saved to evaluation/results.csv")


if __name__ == "__main__":
    main()
