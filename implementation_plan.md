# T10.1 — Passport Assistant: Full Industry-Level Implementation Plan

**Tech Stack:** ChromaDB · LLaMA 3 / Gemini 1.5 Flash · sentence-transformers · Google Translate · Streamlit  
**Type:** Hybrid Agentic + Memory-Based + Multimodal RAG  
**Language Support:** English + Hindi (toggle at start, persistent top-right after)

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Knowledge Base Construction](#3-knowledge-base-construction)
4. [Document Preprocessing & Chunking](#4-document-preprocessing--chunking)
5. [Embedding Pipeline](#5-embedding-pipeline)
6. [ChromaDB Vector Store Setup](#6-chromadb-vector-store-setup)
7. [Retrieval System](#7-retrieval-system)
8. [LLM Integration — Gemini 1.5 Flash](#8-llm-integration--gemini-15-flash)
9. [Memory System](#9-memory-system)
10. [Translation Layer — Hindi/English](#10-translation-layer--hindienglish)
11. [Agentic Routing Layer](#11-agentic-routing-layer)
12. [Streamlit UI — Full Layout & Toggle Logic](#12-streamlit-ui--full-layout--toggle-logic)
13. [Multimodal Input (Image/Document Upload)](#13-multimodal-input-imagedocument-upload)
14. [Evaluation Framework](#14-evaluation-framework)
15. [File-by-File Code Guide](#15-file-by-file-code-guide)
16. [Deployment](#16-deployment)
17. [Submission Checklist](#17-submission-checklist)

---

## 1. Project Structure

```
passport_assistant/
│
├── app.py                        # Streamlit entry point
│
├── config/
│   └── settings.py               # API keys, model names, constants
│
├── data/
│   ├── raw/                      # Original scraped/downloaded docs
│   │   ├── mea_faq.txt
│   │   ├── passport_seva_guide.txt
│   │   ├── tatkal_process.txt
│   │   ├── documents_required.txt
│   │   ├── fees_structure.txt
│   │   ├── annexures_guide.txt
│   │   └── nri_guidelines.txt
│   ├── processed/                # Cleaned, chunked text files
│   │   └── chunks.jsonl          # Final chunks ready for embedding
│   └── evaluation/
│       ├── test_questions_en.json
│       └── test_questions_hi.json
│
├── rag/
│   ├── __init__.py
│   ├── loader.py                 # Load & clean raw documents
│   ├── chunker.py                # Smart chunking logic
│   ├── embedder.py               # Embedding with all-MiniLM-L6-v2
│   ├── vectorstore.py            # ChromaDB setup and operations
│   ├── retriever.py              # Hybrid BM25 + dense retrieval
│   └── generator.py             # Gemini 1.5 Flash integration
│
├── agents/
│   ├── __init__.py
│   ├── router.py                 # Intent classifier → routes to agents
│   ├── eligibility_agent.py      # Am I eligible for passport?
│   ├── document_agent.py         # What documents do I need?
│   ├── fee_agent.py              # Fee calculation
│   ├── appointment_agent.py      # PSK booking process
│   └── status_agent.py          # How to track passport status
│
├── memory/
│   ├── __init__.py
│   └── session_memory.py         # LangChain window memory wrapper
│
├── translation/
│   ├── __init__.py
│   └── translator.py             # Google Translate wrapper (EN↔HI)
│
├── ui/
│   ├── components.py             # Reusable Streamlit UI components
│   ├── language_toggle.py        # Language toggle logic
│   └── chat_interface.py         # Chat bubble rendering
│
├── evaluation/
│   ├── evaluator.py              # RAGAS / custom metric evaluation
│   └── report_generator.py       # Generate evaluation report
│
├── scripts/
│   ├── build_kb.py               # One-shot: scrape + chunk + embed + index
│   └── run_eval.py               # Run full evaluation suite
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_experiments.ipynb
│   └── 03_retrieval_quality.ipynb
│
├── requirements.txt
├── .env                          # API keys (never commit)
├── .gitignore
└── README.md
```

---

## 2. Environment Setup

### 2.1 Python Environment

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# OR
venv\Scripts\activate             # Windows
```

### 2.2 requirements.txt (complete)

```
# Core LLM & RAG
google-generativeai==0.5.4
langchain==0.2.6
langchain-community==0.2.6
langchain-google-genai==1.0.6

# Embeddings
sentence-transformers==3.0.1
torch==2.3.0                      # CPU version is fine for MiniLM

# Vector Store
chromadb==0.5.3

# Retrieval (Hybrid BM25)
rank_bm25==0.2.2

# Translation
googletrans==4.0.0-rc1
deep-translator==1.11.4           # Fallback if googletrans fails

# Document Processing
pypdf==4.2.0
python-docx==1.1.0
beautifulsoup4==4.12.3
requests==2.32.3
lxml==5.2.2

# UI
streamlit==1.36.0
streamlit-chat==0.1.1
Pillow==10.3.0                    # For image uploads

# Evaluation
ragas==0.1.14
pandas==2.2.2
scikit-learn==1.5.0

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
loguru==0.7.2
pydantic==2.7.4
```

```bash
pip install -r requirements.txt
```

### 2.3 .env File

```env
GOOGLE_API_KEY=your_google_ai_studio_key_here
CHROMA_PERSIST_PATH=./chroma_db
COLLECTION_NAME=passport_knowledge_base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-1.5-flash
MAX_RETRIEVED_CHUNKS=5
MEMORY_WINDOW_SIZE=10
DEFAULT_LANGUAGE=en
```

### 2.4 config/settings.py

```python
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "passport_knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", 5))
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", 10))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

# Chunking config
CHUNK_SIZE = 400           # tokens (approx characters / 4)
CHUNK_OVERLAP = 80         # overlap between chunks to preserve context
MIN_CHUNK_SIZE = 100       # discard chunks smaller than this

# System prompt (used in generator)
SYSTEM_PROMPT_EN = """You are a knowledgeable and helpful Passport Assistant for India.
You help users understand passport application processes, document requirements, 
eligibility criteria, fees, and Passport Seva Kendra procedures.

IMPORTANT RULES:
1. Answer ONLY based on the provided context. Do not hallucinate.
2. If the context does not contain the answer, say: "I don't have that information. 
   Please visit passportindia.gov.in or call 1800-258-1800."
3. Always be polite, clear, and concise.
4. Cite the source of information when possible.
5. If user asks in Hindi, respond in Hindi.
"""

SYSTEM_PROMPT_HI = """आप भारत के लिए एक जानकार और सहायक पासपोर्ट सहायक हैं।
आप उपयोगकर्ताओं को पासपोर्ट आवेदन प्रक्रिया, दस्तावेज़ आवश्यकताओं, 
पात्रता मानदंड, शुल्क और पासपोर्ट सेवा केंद्र की प्रक्रियाओं को समझने में मदद करते हैं।

महत्वपूर्ण नियम:
1. केवल दिए गए संदर्भ के आधार पर उत्तर दें। अनुमान न लगाएं।
2. यदि संदर्भ में उत्तर नहीं है, तो कहें: "मेरे पास यह जानकारी नहीं है। 
   कृपया passportindia.gov.in पर जाएं या 1800-258-1800 पर कॉल करें।"
3. हमेशा विनम्र, स्पष्ट और संक्षिप्त रहें।
"""
```

---

## 3. Knowledge Base Construction

### 3.1 Primary Data Sources

Collect data from these official sources (manually download or scrape):

| Source | URL | Content |
|---|---|---|
| Passport Seva Portal | passportindia.gov.in | FAQs, process guides |
| MEA Official | mea.gov.in/passport-services | Policy documents |
| Passport Seva FAQ PDF | passportindia.gov.in/AppOnlineProject/pdf/passport-faq.pdf | Core FAQ |
| Tatkal Guide | passportindia.gov.in (Tatkal section) | Tatkal rules |
| Annexure Templates | passportindia.gov.in/pdf/annexure*.pdf | All annexures |
| NRI Guide | passportindia.gov.in (NRI section) | NRI-specific rules |

### 3.2 Manual Q&A Pairs (curate 100+ yourself)

Create `data/raw/curated_qa.jsonl`:

```jsonl
{"question": "What documents are needed for a fresh passport?", "answer": "For a fresh passport you need: Aadhaar card (proof of identity + address), birth certificate or school certificate (proof of date of birth), and 2 recent passport-size photographs (white background, 3.5x3.5 cm).", "category": "documents", "source": "MEA official"}
{"question": "Tatkal passport kitne din mein milta hai?", "answer": "Tatkal passport generally 1 to 3 working days mein process hota hai police verification ke baad. Emergency cases mein same day bhi possible hai.", "category": "tatkal", "source": "Passport Seva Portal"}
{"question": "What is the fee for a 36-page passport?", "answer": "The fee for a 36-page fresh passport (Normal) is Rs. 1500. For Tatkal, an additional fee of Rs. 2000 is charged, making it Rs. 3500 total.", "category": "fees", "source": "Passport Seva Fee Schedule"}
```

Categories to cover in your Q&A pairs:
- `documents` — what papers are needed
- `eligibility` — who can apply, age rules
- `fees` — charges for all types
- `process` — step-by-step application
- `tatkal` — urgent passport
- `minor` — child passport rules
- `reissue` — renewal, lost, damaged
- `nri` — NRI-specific
- `psk` — Passport Seva Kendra info
- `tracking` — how to track status
- `annexures` — what each annexure is for
- `photo` — photo specifications
- `police_verification` — when and how

### 3.3 Web Scraping Script (scripts/scrape_data.py)

```python
import requests
from bs4 import BeautifulSoup
import time
from pathlib import Path

SOURCES = [
    {
        "url": "https://www.passportindia.gov.in/AppOnlineProject/welcomeLink",
        "name": "passport_seva_home"
    },
    # Add more URLs as needed
]

def scrape_page(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Educational Project)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove nav, footer, scripts
        for tag in soup(["nav", "footer", "script", "style", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Remove excessive blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return ""

Path("data/raw").mkdir(parents=True, exist_ok=True)
for source in SOURCES:
    text = scrape_page(source["url"])
    if text:
        with open(f"data/raw/{source['name']}.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved: {source['name']}.txt ({len(text)} chars)")
    time.sleep(2)  # be polite
```

---

## 4. Document Preprocessing & Chunking

### 4.1 rag/loader.py — Load all raw documents

```python
import json
from pathlib import Path
from loguru import logger
from pypdf import PdfReader

class DocumentLoader:
    def __init__(self, raw_dir: str = "data/raw"):
        self.raw_dir = Path(raw_dir)

    def load_all(self) -> list[dict]:
        """Load all documents from raw directory. Returns list of dicts with text + metadata."""
        docs = []
        for file in self.raw_dir.iterdir():
            if file.suffix == ".txt":
                docs.extend(self._load_txt(file))
            elif file.suffix == ".pdf":
                docs.extend(self._load_pdf(file))
            elif file.suffix == ".jsonl":
                docs.extend(self._load_jsonl(file))
        logger.info(f"Loaded {len(docs)} raw documents from {self.raw_dir}")
        return docs

    def _load_txt(self, path: Path) -> list[dict]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Basic cleaning
        text = self._clean(text)
        return [{"text": text, "source": path.name, "type": "article"}]

    def _load_pdf(self, path: Path) -> list[dict]:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = self._clean(page.extract_text() or "")
            if len(text) > 50:
                pages.append({"text": text, "source": f"{path.name}:page{i+1}", "type": "pdf"})
        return pages

    def _load_jsonl(self, path: Path) -> list[dict]:
        items = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    # Q&A pairs: combine question + answer as one chunk
                    text = f"Q: {obj['question']}\nA: {obj['answer']}"
                    items.append({
                        "text": text,
                        "source": obj.get("source", path.name),
                        "type": "qa",
                        "category": obj.get("category", "general")
                    })
                except:
                    continue
        return items

    def _clean(self, text: str) -> str:
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)      # Collapse excess newlines
        text = re.sub(r'[ \t]{2,}', ' ', text)       # Collapse spaces
        text = text.strip()
        return text
```

### 4.2 rag/chunker.py — Smart sentence-aware chunking

```python
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE
from loguru import logger
import re

class SmartChunker:
    """
    Splits text into overlapping chunks.
    - Tries to break at sentence boundaries (. ! ?)
    - Preserves metadata from original document
    - Skips chunks that are too small
    """

    def __init__(self, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in docs:
            chunks = self._chunk_text(doc["text"])
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "source": doc.get("source", "unknown"),
                    "type": doc.get("type", "article"),
                    "category": doc.get("category", "general"),
                    "chunk_index": i,
                    "chunk_id": f"{doc.get('source','doc')}::chunk{i}"
                })
        logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
        return all_chunks

    def _chunk_text(self, text: str) -> list[str]:
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += " " + sentence
            else:
                if len(current.strip()) >= MIN_CHUNK_SIZE:
                    chunks.append(current.strip())
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current)
                current = overlap_text + " " + sentence

        if len(current.strip()) >= MIN_CHUNK_SIZE:
            chunks.append(current.strip())

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Return last CHUNK_OVERLAP characters of text as overlap."""
        if len(text) <= self.overlap:
            return text
        # Try to start overlap at a sentence boundary
        overlap_region = text[-self.overlap:]
        boundary = overlap_region.find(". ")
        if boundary != -1:
            return overlap_region[boundary+2:]
        return overlap_region
```

---

## 5. Embedding Pipeline

### 5.1 rag/embedder.py

```python
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL
from loguru import logger
import numpy as np

class EmbeddingEngine:
    """
    Wraps sentence-transformers/all-MiniLM-L6-v2.
    - 384-dim embeddings
    - Fully CPU compatible
    - ~80ms per chunk on CPU
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = 384  # MiniLM output dimension
        logger.info("Embedding model loaded successfully")

    def embed_texts(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Embed a list of texts. Returns numpy array of shape (n, 384).
        Uses batching to handle large corpora.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
```

---

## 6. ChromaDB Vector Store Setup

### 6.1 rag/vectorstore.py

```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from config.settings import CHROMA_PERSIST_PATH, COLLECTION_NAME
from rag.embedder import EmbeddingEngine
from loguru import logger
from typing import Optional
import uuid

class PassportVectorStore:
    """
    Manages the ChromaDB collection for passport knowledge base.
    - Persistent storage (survives restarts)
    - Custom embedding function using our MiniLM model
    - Metadata filtering by category and source
    """

    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_PATH,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance
        )
        logger.info(f"ChromaDB connected. Collection '{COLLECTION_NAME}' has "
                    f"{self.collection.count()} documents.")

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add processed chunks to ChromaDB. Skips if already indexed."""
        if self.collection.count() > 0:
            logger.warning("Collection already has data. Skipping re-indexing. "
                           "Delete chroma_db/ folder to re-index from scratch.")
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)

        ids = [c["chunk_id"] if "chunk_id" in c else str(uuid.uuid4()) for c in chunks]
        metadatas = [
            {
                "source": c.get("source", "unknown"),
                "type": c.get("type", "article"),
                "category": c.get("category", "general"),
                "chunk_index": str(c.get("chunk_index", 0))
            }
            for c in chunks
        ]

        # ChromaDB requires embeddings as list of lists
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        logger.success(f"Indexed {len(chunks)} chunks into ChromaDB.")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        category_filter: Optional[str] = None
    ) -> list[dict]:
        """
        Query the vector store.
        Returns list of dicts with: text, source, category, distance.
        """
        query_embedding = self.embedder.embed_query(query_text)
        where_filter = {"category": category_filter} if category_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "category": results["metadatas"][0][i]["category"],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i]  # cosine → similarity
            })
        return retrieved

    def get_collection_stats(self) -> dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": COLLECTION_NAME
        }
```

---

## 7. Retrieval System

### 7.1 rag/retriever.py — Hybrid BM25 + Dense

```python
from rank_bm25 import BM25Okapi
from rag.vectorstore import PassportVectorStore
from config.settings import MAX_RETRIEVED_CHUNKS
from loguru import logger
import re

class HybridRetriever:
    """
    Combines:
    - Dense retrieval: ChromaDB + MiniLM embeddings (semantic understanding)
    - Sparse retrieval: BM25 (exact keyword matching — great for rule numbers, form names)
    
    Final ranking: weighted combination (alpha = dense weight).
    """

    def __init__(self, vectorstore: PassportVectorStore, all_chunks: list[dict], alpha: float = 0.7):
        self.vectorstore = vectorstore
        self.alpha = alpha          # weight for dense retrieval
        self.bm25_alpha = 1 - alpha # weight for BM25
        self._build_bm25(all_chunks)

    def _build_bm25(self, chunks: list[dict]) -> None:
        """Build BM25 index from all chunks."""
        self.bm25_corpus = [c["text"] for c in chunks]
        tokenized = [self._tokenize(text) for text in self.bm25_corpus]
        self.bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built with {len(self.bm25_corpus)} documents.")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer: lowercase + split on non-alphanumeric."""
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def retrieve(self, query: str, n_results: int = MAX_RETRIEVED_CHUNKS) -> list[dict]:
        """Retrieve top-n chunks using hybrid scoring."""
        # --- Dense retrieval ---
        dense_results = self.vectorstore.query(query, n_results=n_results * 2)
        dense_scores = {r["text"][:80]: r["relevance_score"] for r in dense_results}

        # --- BM25 retrieval ---
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]

        # --- Combine scores ---
        combined = {}
        # Add dense results
        for r in dense_results:
            key = r["text"][:80]
            combined[key] = {
                "text": r["text"],
                "source": r["source"],
                "category": r["category"],
                "score": self.alpha * dense_scores.get(key, 0)
            }
        # Add BM25 contribution
        for i, (text, score) in enumerate(zip(self.bm25_corpus, bm25_norm)):
            key = text[:80]
            if key in combined:
                combined[key]["score"] += self.bm25_alpha * score
            elif score > 0.1:  # Only add BM25-only results if they're somewhat relevant
                combined[key] = {
                    "text": text,
                    "source": "bm25_match",
                    "category": "general",
                    "score": self.bm25_alpha * score
                }

        # Sort by combined score, return top n
        sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:n_results]
```

---

## 8. LLM Integration — Gemini 1.5 Flash

### 8.1 rag/generator.py

```python
import google.generativeai as genai
from config.settings import GOOGLE_API_KEY, LLM_MODEL, SYSTEM_PROMPT_EN, SYSTEM_PROMPT_HI
from loguru import logger

genai.configure(api_key=GOOGLE_API_KEY)

class ResponseGenerator:
    """
    Generates final answers using Gemini 1.5 Flash.
    - Grounds responses in retrieved context
    - Handles both English and Hindi
    - Adds source citations
    - Falls back gracefully when context is insufficient
    """

    def __init__(self, language: str = "en"):
        self.model = genai.GenerativeModel(LLM_MODEL)
        self.language = language

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        conversation_history: list[dict] = None
    ) -> dict:
        """
        Generate a response.
        Returns: {"answer": str, "sources": list[str], "confidence": str}
        """
        system_prompt = SYSTEM_PROMPT_HI if self.language == "hi" else SYSTEM_PROMPT_EN

        # Build context string from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Build conversation history string
        history_str = ""
        if conversation_history:
            for turn in conversation_history[-4:]:  # Last 4 turns
                history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        # Build final prompt
        prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{history_str}

RETRIEVED CONTEXT:
{context}

CURRENT QUESTION: {query}

Provide a clear, helpful answer based ONLY on the context above.
At the end, mention which source(s) you used in format: [Source: ...]
If you cannot answer from context, say so clearly.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,       # Low temp for factual accuracy
                    max_output_tokens=1024,
                    top_p=0.95
                )
            )
            answer = response.text
            sources = [c["source"] for c in retrieved_chunks[:3]]
            confidence = self._assess_confidence(retrieved_chunks)

            return {
                "answer": answer,
                "sources": list(set(sources)),
                "confidence": confidence,
                "retrieved_count": len(retrieved_chunks)
            }

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            fallback = ("मैं अभी इस प्रश्न का उत्तर देने में असमर्थ हूं। कृपया passportindia.gov.in पर जाएं।"
                       if self.language == "hi" else
                       "I'm unable to answer that right now. Please visit passportindia.gov.in or call 1800-258-1800.")
            return {"answer": fallback, "sources": [], "confidence": "low"}

    def _build_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[{i}] (Source: {chunk['source']})\n{chunk['text']}")
        return "\n\n---\n\n".join(context_parts)

    def _assess_confidence(self, chunks: list[dict]) -> str:
        if not chunks:
            return "low"
        avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
        if avg_score > 0.75:
            return "high"
        elif avg_score > 0.5:
            return "medium"
        return "low"
```

---

## 9. Memory System

### 9.1 memory/session_memory.py

```python
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Turn:
    user: str
    assistant: str
    language: str = "en"
    intent: Optional[str] = None

class SessionMemory:
    """
    Sliding window conversation memory.
    - Keeps last N turns in memory
    - Provides formatted history for LLM context
    - Tracks language preference per session
    - Stores detected intents for routing
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.turns: deque[Turn] = deque(maxlen=window_size)
        self.session_language: str = "en"
        self.user_profile: dict = {}   # Can store: name, application_type, etc.

    def add_turn(self, user_msg: str, assistant_msg: str, language: str = "en", intent: str = None):
        self.turns.append(Turn(
            user=user_msg,
            assistant=assistant_msg,
            language=language,
            intent=intent
        ))

    def get_history(self) -> list[dict]:
        return [{"user": t.user, "assistant": t.assistant} for t in self.turns]

    def get_last_n(self, n: int = 3) -> list[dict]:
        recent = list(self.turns)[-n:]
        return [{"user": t.user, "assistant": t.assistant} for t in recent]

    def get_context_summary(self) -> str:
        """Build a short context summary for the LLM."""
        if not self.turns:
            return "This is the start of the conversation."
        recent_intents = [t.intent for t in self.turns if t.intent]
        topics = list(set(recent_intents[-3:]))
        return f"Recent topics discussed: {', '.join(topics) if topics else 'general inquiry'}"

    def clear(self):
        self.turns.clear()
        self.user_profile.clear()

    def update_user_profile(self, key: str, value):
        """Store user-specific info learned during conversation."""
        self.user_profile[key] = value

    def get_user_profile(self) -> dict:
        return self.user_profile
```

---

## 10. Translation Layer — Hindi/English

### 10.1 translation/translator.py

```python
from deep_translator import GoogleTranslator
from loguru import logger
import re

class BilingualTranslator:
    """
    Handles Hindi ↔ English translation.
    - Detects language of input
    - Translates query to English for retrieval (embeddings are English-trained)
    - Translates response back to Hindi if needed
    - Uses deep-translator (more reliable than googletrans)
    """

    HINDI_CHARS = re.compile(r'[\u0900-\u097F]')  # Devanagari Unicode block

    def __init__(self):
        self.en_to_hi = GoogleTranslator(source='en', target='hi')
        self.hi_to_en = GoogleTranslator(source='hi', target='en')

    def detect_language(self, text: str) -> str:
        """Returns 'hi' if text contains Hindi characters, else 'en'."""
        hindi_count = len(self.HINDI_CHARS.findall(text))
        total_chars = len(text.replace(" ", ""))
        if total_chars == 0:
            return "en"
        ratio = hindi_count / total_chars
        return "hi" if ratio > 0.15 else "en"

    def to_english(self, text: str) -> str:
        """Translate Hindi text to English for embedding lookup."""
        try:
            detected = self.detect_language(text)
            if detected == "hi":
                translated = self.hi_to_en.translate(text)
                logger.debug(f"HI→EN: '{text[:50]}' → '{translated[:50]}'")
                return translated
            return text  # Already English
        except Exception as e:
            logger.warning(f"Translation HI→EN failed: {e}. Using original text.")
            return text

    def to_hindi(self, text: str) -> str:
        """Translate English text to Hindi for response."""
        try:
            translated = self.en_to_hi.translate(text)
            return translated
        except Exception as e:
            logger.warning(f"Translation EN→HI failed: {e}. Returning English.")
            return text

    def translate_query_for_retrieval(self, query: str, ui_language: str) -> str:
        """
        Always return an English version of the query for vector search.
        Embeddings (MiniLM) are trained on English — retrieval quality is better in English.
        """
        detected = self.detect_language(query)
        if detected == "hi" or ui_language == "hi":
            return self.to_english(query)
        return query
```

---

## 11. Agentic Routing Layer

### 11.1 agents/router.py — Intent Classification

```python
import re
from loguru import logger

# Intent → keywords mapping
INTENT_PATTERNS = {
    "eligibility": [
        r"eligible", r"can i apply", r"qualify", r"age limit",
        r"nationality", r"who can", r"पात्र", r"आवेदन कर सकते"
    ],
    "documents": [
        r"document", r"papers", r"proof", r"certificate", r"aadhaar",
        r"birth cert", r"what do i need", r"दस्तावेज़", r"कागज़"
    ],
    "fees": [
        r"fee", r"cost", r"price", r"charge", r"how much", r"amount",
        r"payment", r"शुल्क", r"कितना", r"पैसे"
    ],
    "tatkal": [
        r"tatkal", r"urgent", r"emergency", r"fast", r"quick",
        r"तत्काल", r"जल्दी"
    ],
    "appointment": [
        r"appointment", r"book", r"slot", r"psk", r"passport seva",
        r"visit", r"नियुक्ति", r"बुकिंग"
    ],
    "tracking": [
        r"status", r"track", r"where", r"dispatch", r"delivered",
        r"स्थिति", r"ट्रैक"
    ],
    "reissue": [
        r"renew", r"reissue", r"expired", r"lost", r"damage",
        r"नवीनीकरण", r"खो गया"
    ],
    "minor": [
        r"child", r"minor", r"kid", r"son", r"daughter", r"baby",
        r"बच्चे", r"नाबालिग"
    ],
    "nri": [
        r"nri", r"abroad", r"foreign", r"overseas", r"outside india",
        r"प्रवासी"
    ],
    "photo": [
        r"photo", r"photograph", r"picture", r"size", r"specification",
        r"फोटो"
    ],
    "general": []  # fallback
}

class IntentRouter:
    """
    Classifies user query into one of the passport intents.
    Uses regex pattern matching (fast, no model needed).
    """

    def classify(self, query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for intent, patterns in INTENT_PATTERNS.items():
            if intent == "general":
                continue
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general"
        best_intent = max(scores, key=scores.get)
        logger.debug(f"Intent '{best_intent}' detected for query: {query[:50]}")
        return best_intent

    def get_category_filter(self, intent: str) -> str | None:
        """Map intent to ChromaDB category filter for targeted retrieval."""
        mapping = {
            "eligibility": "eligibility",
            "documents": "documents",
            "fees": "fees",
            "tatkal": "tatkal",
            "appointment": "process",
            "tracking": "process",
            "reissue": "reissue",
            "minor": "minor",
            "nri": "nri",
            "photo": "documents",
            "general": None  # No filter → search all
        }
        return mapping.get(intent, None)
```

---

## 12. Streamlit UI — Full Layout & Toggle Logic

### 12.1 ui/language_toggle.py

```python
import streamlit as st

UI_TEXT = {
    "en": {
        "title": "🛂 Passport Assistant",
        "subtitle": "Your guide to Indian passport services",
        "welcome": "Hello! I'm your Passport Assistant. Ask me anything about passport applications, documents, fees, or procedures.",
        "input_placeholder": "Type your question here...",
        "send_button": "Send",
        "clear_button": "Clear Chat",
        "lang_toggle_label": "🌐 Language",
        "sources_label": "📚 Sources",
        "confidence_label": "Confidence",
        "language_select": "Select Language / भाषा चुनें",
        "thinking": "Thinking...",
        "upload_label": "📎 Upload Document (optional)",
    },
    "hi": {
        "title": "🛂 पासपोर्ट सहायक",
        "subtitle": "भारतीय पासपोर्ट सेवाओं के लिए आपका मार्गदर्शक",
        "welcome": "नमस्ते! मैं आपका पासपोर्ट सहायक हूं। पासपोर्ट आवेदन, दस्तावेज़, शुल्क या प्रक्रियाओं के बारे में कुछ भी पूछें।",
        "input_placeholder": "अपना प्रश्न यहां टाइप करें...",
        "send_button": "भेजें",
        "clear_button": "चैट साफ़ करें",
        "lang_toggle_label": "🌐 भाषा",
        "sources_label": "📚 स्रोत",
        "confidence_label": "विश्वसनीयता",
        "language_select": "Select Language / भाषा चुनें",
        "thinking": "सोच रहा हूं...",
        "upload_label": "📎 दस्तावेज़ अपलोड करें (वैकल्पिक)",
    }
}

def get_ui_text(key: str) -> str:
    lang = st.session_state.get("language", "en")
    return UI_TEXT.get(lang, UI_TEXT["en"]).get(key, key)

def render_language_selector_startup():
    """
    Full-page language selector shown ONCE at startup.
    After selection, redirects to main chat.
    """
    st.markdown("""
    <div style='text-align: center; padding: 80px 0 40px 0;'>
        <h1>🛂 Passport Assistant</h1>
        <h1>🛂 पासपोर्ट सहायक</h1>
        <p style='font-size: 1.2em; color: #666;'>
            Please select your preferred language<br>
            कृपया अपनी पसंदीदा भाषा चुनें
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lang_choice = st.radio(
            "Language / भाषा",
            options=["English", "हिंदी"],
            horizontal=True,
            label_visibility="collapsed"
        )
        st.write("")
        if st.button("Continue / जारी रखें", use_container_width=True, type="primary"):
            st.session_state["language"] = "hi" if lang_choice == "हिंदी" else "en"
            st.session_state["language_selected"] = True
            st.rerun()

def render_language_toggle_topright():
    """
    Compact toggle shown in top-right corner after initial selection.
    """
    current = st.session_state.get("language", "en")
    new_lang = "hi" if current == "en" else "en"
    label = "हिंदी" if current == "en" else "English"

    with st.container():
        _, col_right = st.columns([6, 1])
        with col_right:
            if st.button(f"🌐 {label}", key="lang_toggle_btn"):
                st.session_state["language"] = new_lang
                st.rerun()
```

### 12.2 app.py — Main Streamlit App

```python
import streamlit as st
import sys
from pathlib import Path

# --- Path setup ---
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DEFAULT_LANGUAGE, MAX_RETRIEVED_CHUNKS
from rag.loader import DocumentLoader
from rag.chunker import SmartChunker
from rag.embedder import EmbeddingEngine
from rag.vectorstore import PassportVectorStore
from rag.retriever import HybridRetriever
from rag.generator import ResponseGenerator
from memory.session_memory import SessionMemory
from translation.translator import BilingualTranslator
from agents.router import IntentRouter
from ui.language_toggle import render_language_selector_startup, render_language_toggle_topright, get_ui_text

# ==== Page config ====
st.set_page_config(
    page_title="Passport Assistant | पासपोर्ट सहायक",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==== Custom CSS ====
st.markdown("""
<style>
    /* Chat bubbles */
    .user-bubble {
        background: #0066CC;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 20%;
        text-align: right;
    }
    .bot-bubble {
        background: #F0F2F6;
        color: #1a1a1a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 20% 8px 0;
    }
    .source-tag {
        font-size: 0.75em;
        color: #888;
        margin-top: 4px;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Language toggle alignment */
    .lang-toggle { position: fixed; top: 60px; right: 20px; z-index: 999; }
</style>
""", unsafe_allow_html=True)

# ==== Session State Init ====
if "language_selected" not in st.session_state:
    st.session_state["language_selected"] = False
if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # List of (role, text) tuples
if "memory" not in st.session_state:
    st.session_state["memory"] = SessionMemory()

# ==== Step 1: Show language selector if not yet selected ====
if not st.session_state["language_selected"]:
    render_language_selector_startup()
    st.stop()

# ==== Step 2: Load RAG components (cached so they load once) ====
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag_pipeline():
    loader = DocumentLoader("data/raw")
    chunker = SmartChunker()
    embedder = EmbeddingEngine()
    docs = loader.load_all()
    chunks = chunker.chunk_documents(docs)
    vectorstore = PassportVectorStore(embedder)
    vectorstore.add_chunks(chunks)
    retriever = HybridRetriever(vectorstore, chunks)
    return retriever, vectorstore

retriever, vectorstore = load_rag_pipeline()
translator = BilingualTranslator()
router = IntentRouter()

# ==== Step 3: Main Chat Interface ====
# Language toggle — top right
render_language_toggle_topright()

lang = st.session_state["language"]
generator = ResponseGenerator(language=lang)

# Header
st.title(get_ui_text("title"))
st.caption(get_ui_text("subtitle"))

# Stats in sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    stats = vectorstore.get_collection_stats()
    st.metric("Knowledge Chunks", stats["total_chunks"])
    st.metric("Language", "Hindi 🇮🇳" if lang == "hi" else "English 🇬🇧")
    if st.button(get_ui_text("clear_button")):
        st.session_state["chat_history"] = []
        st.session_state["memory"].clear()
        st.rerun()
    st.divider()
    st.markdown("**Quick Topics:**")
    for topic in ["Fresh Passport", "Tatkal", "Renewal", "Documents", "Fees"]:
        if st.button(topic, use_container_width=True):
            st.session_state["quick_query"] = topic

# Welcome message
if not st.session_state["chat_history"]:
    st.info(get_ui_text("welcome"))

# Chat history display
chat_container = st.container()
with chat_container:
    for role, text in st.session_state["chat_history"]:
        if role == "user":
            st.markdown(f'<div class="user-bubble">👤 {text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">🛂 {text}</div>', unsafe_allow_html=True)

# Document upload (optional)
with st.expander(get_ui_text("upload_label")):
    uploaded_file = st.file_uploader("Upload Aadhaar, Birth Certificate, etc.", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name} — I'll use this context in my answers.")

# Chat input
st.divider()
col_input, col_btn = st.columns([5, 1])
with col_input:
    # Handle quick query from sidebar
    default_val = st.session_state.pop("quick_query", "")
    user_input = st.text_input(
        get_ui_text("input_placeholder"),
        value=default_val,
        label_visibility="collapsed",
        key="user_input_box"
    )
with col_btn:
    send_clicked = st.button(get_ui_text("send_button"), type="primary", use_container_width=True)

# Process query
if (send_clicked or user_input) and user_input.strip():
    query = user_input.strip()

    # Add user message to UI history
    st.session_state["chat_history"].append(("user", query))

    with st.spinner(get_ui_text("thinking")):
        # Translate query to English for retrieval
        english_query = translator.translate_query_for_retrieval(query, lang)

        # Classify intent
        intent = router.classify(english_query)
        category_filter = router.get_category_filter(intent)

        # Retrieve relevant chunks
        chunks = retriever.retrieve(english_query, n_results=MAX_RETRIEVED_CHUNKS)

        # Get conversation history
        history = st.session_state["memory"].get_history()

        # Generate response
        result = generator.generate(english_query, chunks, history)
        answer = result["answer"]

        # If UI language is Hindi but answer came back in English, translate
        if lang == "hi" and translator.detect_language(answer) == "en":
            answer = translator.to_hindi(answer)

        # Add to memory
        st.session_state["memory"].add_turn(query, answer, lang, intent)

        # Add to chat history
        display_answer = answer
        if result["sources"]:
            display_answer += f"\n\n*Sources: {', '.join(result['sources'][:2])}*"

        st.session_state["chat_history"].append(("assistant", display_answer))

    st.rerun()
```

---

## 13. Multimodal Input (Image/Document Upload)

### 13.1 How to handle uploaded passport photos or documents

```python
# In app.py — after file upload
from PIL import Image
import io

def process_uploaded_document(uploaded_file) -> str:
    """
    Extract text or analyze image from uploaded file.
    Returns a text summary to inject into the RAG context.
    """
    file_type = uploaded_file.type

    if "pdf" in file_type:
        # Extract text from PDF
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        return f"[USER UPLOADED DOCUMENT — Content]: {text[:1000]}"

    elif "image" in file_type:
        # Send image to Gemini Vision for analysis
        import google.generativeai as genai
        from config.settings import GOOGLE_API_KEY, LLM_MODEL
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(LLM_MODEL)
        image = Image.open(uploaded_file)
        response = model.generate_content([
            "This is a document related to a passport application. "
            "Extract and summarize all visible text and key information from this image. "
            "Focus on: name, date of birth, address, document type.",
            image
        ])
        return f"[USER UPLOADED IMAGE — Extracted Info]: {response.text}"

    return ""
```

---

## 14. Evaluation Framework

### 14.1 evaluation/evaluator.py

```python
import json
from rag.retriever import HybridRetriever
from rag.generator import ResponseGenerator
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
        with open(path) as f:
            return json.load(f)  # [{"question": ..., "answer": ..., "category": ...}]

    def evaluate(self, test_set: list[dict]) -> pd.DataFrame:
        results = []
        for item in test_set:
            import time
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
        import re
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if not sentences or not chunks:
            return 0.0
        context_text = " ".join(c["text"] for c in chunks)
        supported = 0
        for s in sentences:
            if len(s) < 20:
                continue
            sim = self._semantic_similarity(s, context_text)
            if sim > 0.5:
                supported += 1
        return supported / max(len(sentences), 1)
```

### 14.2 Evaluation Test Set Format (data/evaluation/test_questions_en.json)

```json
[
  {
    "question": "What documents are required for a fresh passport application?",
    "answer": "For a fresh passport you need: Aadhaar card, birth certificate or school leaving certificate, 2 passport-size photographs with white background, and proof of address.",
    "category": "documents"
  },
  {
    "question": "What is the fee for Tatkal passport?",
    "answer": "The Tatkal fee is Rs. 2000 in addition to the normal fee of Rs. 1500 for a 36-page booklet, making total Rs. 3500.",
    "category": "fees"
  }
]
```

---

## 15. File-by-File Code Guide

### 15.1 scripts/build_kb.py — One-shot knowledge base builder

```python
"""
Run this ONCE to scrape, chunk, embed, and index all documents.
Usage: python scripts/build_kb.py
"""
import sys
sys.path.insert(0, ".")

from rag.loader import DocumentLoader
from rag.chunker import SmartChunker
from rag.embedder import EmbeddingEngine
from rag.vectorstore import PassportVectorStore
from loguru import logger
import json

def main():
    logger.info("=== Building Passport Knowledge Base ===")
    loader = DocumentLoader("data/raw")
    chunker = SmartChunker()
    embedder = EmbeddingEngine()

    logger.info("Step 1: Loading documents...")
    docs = loader.load_all()
    logger.info(f"Loaded {len(docs)} documents")

    logger.info("Step 2: Chunking...")
    chunks = chunker.chunk_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    # Save chunks to disk for inspection
    with open("data/processed/chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info("Step 3: Indexing into ChromaDB...")
    vectorstore = PassportVectorStore(embedder)
    vectorstore.add_chunks(chunks)

    logger.success("Knowledge base built successfully!")
    logger.info(f"Stats: {vectorstore.get_collection_stats()}")

if __name__ == "__main__":
    main()
```

### 15.2 scripts/run_eval.py

```python
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
import json

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
    df.to_csv("evaluation/results.csv", index=False)
    print("Results saved to evaluation/results.csv")

if __name__ == "__main__":
    main()
```

---

## 16. Deployment

### 16.1 Streamlit Cloud (Free)

1. Push project to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set secrets in Streamlit Cloud dashboard:
   ```
   GOOGLE_API_KEY = "your_key_here"
   ```
5. Set main file: `app.py`
6. Deploy — public URL generated automatically

### 16.2 Hugging Face Spaces (Alternative Free Option)

```
# Create a Space with Streamlit SDK
# Upload all files
# Add GOOGLE_API_KEY in Space Secrets
```

### 16.3 .gitignore

```
.env
chroma_db/
venv/
__pycache__/
*.pyc
data/raw/*.pdf
.DS_Store
```

### 16.4 README.md (for submission)

```markdown
# T10.1 — Passport Assistant

A bilingual (Hindi/English) RAG-based conversational AI for Indian passport services.

## Setup
1. Clone repo
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY`
4. Add documents to `data/raw/`
5. `python scripts/build_kb.py` — builds vector DB
6. `streamlit run app.py` — launch app

## Architecture
- **Retrieval**: Hybrid BM25 + Dense (ChromaDB + MiniLM)
- **Generation**: Gemini 1.5 Flash
- **Memory**: Sliding window session memory
- **Translation**: Google Translate API (Hindi ↔ English)
- **Agents**: Intent-based routing to specialized sub-agents

## Evaluation
Run: `python scripts/run_eval.py`
Results saved to `evaluation/results.csv`
```

---

## 17. Submission Checklist

### Code
- [ ] `app.py` — working Streamlit app
- [ ] `rag/` — all RAG components implemented
- [ ] `agents/` — router + at least 3 agents
- [ ] `memory/` — session memory working
- [ ] `translation/` — Hindi/English toggle working
- [ ] `evaluation/` — evaluator with results
- [ ] `scripts/build_kb.py` — one-shot KB builder

### Data
- [ ] At least 7 raw documents in `data/raw/`
- [ ] At least 50 curated Q&A pairs in `curated_qa.jsonl`
- [ ] Test set with 30+ questions in `data/evaluation/`

### Features
- [ ] Language toggle at startup (full-page)
- [ ] Language toggle moves to top-right after selection
- [ ] Hindi interface fully functional
- [ ] Conversation memory (follow-up questions work)
- [ ] Source citations in responses
- [ ] "I don't know" fallback when context is insufficient
- [ ] Document/image upload (optional but good)

### Evaluation Metrics to Report
- [ ] Retrieval Recall@5 (are relevant chunks retrieved?)
- [ ] Answer Correctness (semantic similarity to ground truth)
- [ ] Answer Faithfulness (grounded in context?)
- [ ] Mean latency per query
- [ ] Results broken down by category

### Report
- [ ] System architecture diagram
- [ ] Chunking strategy justification
- [ ] Embedding model choice justification
- [ ] Hybrid retrieval explanation
- [ ] Sample conversation screenshots
- [ ] Evaluation results table
- [ ] Challenges faced + future work

---

*Implementation Plan v1.0 — T10.1 Passport Assistant | SMAI A3*
