import re
from rank_bm25 import BM25Okapi
from rag.vectorstore import PassportVectorStore
from config.settings import MAX_RETRIEVED_CHUNKS


class HybridRetriever:
    def __init__(self, vectorstore: PassportVectorStore, all_chunks: list, alpha: float = 0.7):
        self.vectorstore = vectorstore
        self.alpha = alpha
        self.bm25_alpha = 1 - alpha
        self.bm25_corpus = [c["text"] for c in all_chunks]
        self.bm25_metadata = all_chunks
        tokenized = [self._tokenize(t) for t in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized)
        print(f"[INFO] BM25 index built with {len(self.bm25_corpus)} docs.")

    def _tokenize(self, text: str) -> list:
        return re.findall(r'\b\w+\b', text.lower())

    def retrieve(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> list:
        dense_results = self.vectorstore.query(query, n_results=n_results * 2)

        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        max_bm25 = float(max(bm25_scores)) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_norm = [float(s) / max_bm25 for s in bm25_scores]

        combined = {}
        for r in dense_results:
            key = hash(r["text"])
            combined[key] = {
                "text": r["text"],
                "source": r["source"],
                "category": r["category"],
                "score": self.alpha * r["relevance_score"]
            }
        for i, (text, score) in enumerate(zip(self.bm25_corpus, bm25_norm)):
            if score < 0.05:
                continue
            key = hash(text)
            meta = self.bm25_metadata[i] if i < len(self.bm25_metadata) else {}
            if key in combined:
                combined[key]["score"] += self.bm25_alpha * score
            else:
                combined[key] = {
                    "text": text,
                    "source": meta.get("source", "bm25_match"),
                    "category": meta.get("category", "general"),
                    "score": self.bm25_alpha * score
                }

        return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:n_results]
