import os
import uuid
from typing import Optional

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"

try:
    import chromadb.telemetry.opentelemetry as _chroma_otel
    _chroma_otel.otel_init = lambda *a, **kw: None
except Exception:
    pass

import chromadb
from config.settings import CHROMA_PERSIST_PATH, COLLECTION_NAME
from rag.embedder import EmbeddingEngine


class PassportVectorStore:
    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_PATH,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"[INFO] ChromaDB ready. '{COLLECTION_NAME}' has {self.collection.count()} docs.")

    def add_chunks(self, chunks: list) -> None:
        if self.collection.count() > 0:
            print("[WARNING] Collection already indexed. Delete chroma_db/ to re-index.")
            return
        if not chunks:
            print("[ERROR] No chunks to index.")
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids = [str(c.get("chunk_id", uuid.uuid4()))[:512] for c in chunks]
        metadatas = [
            {
                "source": str(c.get("source", "unknown")),
                "type": str(c.get("type", "article")),
                "category": str(c.get("category", "general")),
                "chunk_index": str(c.get("chunk_index", 0))
            }
            for c in chunks
        ]

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            print(f"[INFO] Indexed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
        print("[INFO] Indexing complete.")

    def query(self, query_text: str, n_results: int = 5, category_filter: Optional[str] = None) -> list:
        total = self.collection.count()
        if total == 0:
            return []
        n_results = min(n_results, total)
        query_embedding = self.embedder.embed_query(query_text)
        where_filter = {"category": category_filter} if category_filter and category_filter != "general" else None

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "category": results["metadatas"][0][i]["category"],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i]
            })
        return retrieved

    def get_collection_stats(self) -> dict:
        return {"total_chunks": self.collection.count(), "collection_name": COLLECTION_NAME}
