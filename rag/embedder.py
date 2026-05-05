import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


class EmbeddingEngine:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = 384
        print("[INFO] Embedding model loaded.")

    def embed_texts(self, texts: list, batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
