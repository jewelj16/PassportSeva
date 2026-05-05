import re
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


class SmartChunker:
    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, docs: list) -> list:
        all_chunks = []
        for doc_idx, doc in enumerate(docs):
            chunks = self._chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                safe_source = re.sub(r'[^a-zA-Z0-9_\-.]', '_', doc.get("source", "doc"))
                all_chunks.append({
                    "text": chunk,
                    "source": doc.get("source", "unknown"),
                    "type": doc.get("type", "article"),
                    "category": doc.get("category", "general"),
                    "chunk_index": i,
                    "chunk_id": f"doc{doc_idx:04d}_chunk{i:04d}_{safe_source[:40]}"
                })
        print(f"[INFO] Created {len(all_chunks)} chunks from {len(docs)} documents")
        return all_chunks

    def _chunk_text(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
            if len(current) + len(sentence) <= self.chunk_size:
                current = (current + " " + sentence).strip()
            else:
                if len(current) >= MIN_CHUNK_SIZE:
                    chunks.append(current)
                overlap_text = self._get_overlap(current)
                current = (overlap_text + " " + sentence).strip()
        if len(current) >= MIN_CHUNK_SIZE:
            chunks.append(current)
        return chunks

    def _get_overlap(self, text: str) -> str:
        if len(text) <= self.overlap:
            return text
        overlap_region = text[-self.overlap:]
        boundary = overlap_region.find(". ")
        if boundary != -1:
            return overlap_region[boundary + 2:]
        return overlap_region
