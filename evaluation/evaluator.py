import json
import time
import re
from rag.retriever import HybridRetriever
from rag.generator1 import ResponseGenerator
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd


class PassportRAGEvaluator:
    """
    Evaluates the RAG system on:
    1. Retrieval Recall@K — are relevant chunks retrieved?
    2. Answer Faithfulness — does answer come from retrieved context?
    3. Answer Correctness — semantic similarity to ground truth
    4. Latency — end-to-end response time
    """

    def __init__(self, retriever: HybridRetriever, generator: ResponseGenerator):
        self.retriever = retriever
        self.generator = generator
        self.sem_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def load_test_set(self, path: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def evaluate(self, test_set: list[dict]) -> pd.DataFrame:
        results = []
        for item in test_set:
            start = time.time()
            chunks = self.retriever.retrieve(item["question"])
            result = self.generator.generate(item["question"], chunks)
            latency = time.time() - start

            correctness = self._semantic_similarity(result["answer"], item["answer"])
            faithfulness = self._check_faithfulness(result["answer"], chunks)

            results.append({
                "question": item["question"],
                "expected": item["answer"],
                "generated": result["answer"],
                "category": item.get("category", "general"),
                "correctness": correctness,
                "faithfulness": faithfulness,
                "latency_s": round(latency, 2),
                "confidence": result["confidence"],
                "chunks_retrieved": len(chunks)
            })

        df = pd.DataFrame(results)
        print("\n===== EVALUATION RESULTS =====")
        print(f"Mean Correctness:  {df['correctness'].mean():.3f}")
        print(f"Mean Faithfulness: {df['faithfulness'].mean():.3f}")
        print(f"Mean Latency:      {df['latency_s'].mean():.2f}s")
        print(f"High Confidence:   {(df['confidence']=='high').mean()*100:.1f}%")
        return df

    def _semantic_similarity(self, a: str, b: str) -> float:
        emb = self.sem_model.encode([a, b], normalize_embeddings=True)
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

    def _check_faithfulness(self, answer: str, chunks: list[dict]) -> float:
        """Check what fraction of answer sentences are supported by context."""
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if not sentences or not chunks:
            return 0.0
        context_text = " ".join(c["text"] for c in chunks)
        supported = 0
        total = 0
        for s in sentences:
            if len(s) < 20:
                continue
            total += 1
            sim = self._semantic_similarity(s, context_text)
            if sim > 0.5:
                supported += 1
        return supported / max(total, 1)
