# T10.1 — Passport Assistant 🛂

A bilingual (Hindi/English) RAG-based conversational AI for Indian passport services.

## Architecture

- **Retrieval**: Hybrid BM25 + Dense (ChromaDB + MiniLM)
- **Generation**: Gemini 1.5 Flash
- **Memory**: Sliding window session memory
- **Translation**: Google Translate API (Hindi ↔ English)
- **Agents**: Intent-based routing to specialized sub-agents
- **UI**: Streamlit with custom chat interface

## Setup

1. Clone this repo
2. Create a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env` and fill in your Google AI Studio API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
5. Build the knowledge base (one-time):
   ```bash
   python scripts/build_kb.py
   ```
6. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
passport_assistant/
├── app.py                    # Streamlit entry point
├── config/settings.py        # Configuration & system prompts
├── data/
│   ├── raw/                  # 7 knowledge base documents + curated Q&A
│   ├── processed/            # Chunked data
│   └── evaluation/           # Test sets (EN + HI)
├── rag/                      # RAG pipeline (loader, chunker, embedder, vectorstore, retriever, generator)
├── agents/                   # Intent router
├── memory/                   # Session memory
├── translation/              # Hindi ↔ English translation
├── ui/                       # Streamlit UI components
├── evaluation/               # Evaluation framework
└── scripts/                  # Build KB & run evaluation
```

## Evaluation

```bash
python scripts/run_eval.py
```
Results saved to `evaluation/results.csv`.

## API Key Required

You need a **Google AI Studio API key** (free) for the Gemini 1.5 Flash LLM.
Get it at: https://aistudio.google.com/app/apikey

## Team

SMAI Assignment 3 — Topic T10.1

#env file format: 
```
GOOGLE_API_KEY=<your_google_api_key>
CHROMA_PERSIST_PATH=./chroma_db
COLLECTION_NAME=passport_knowledge_base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-1.5-flash
MAX_RETRIEVED_CHUNKS=5
MEMORY_WINDOW_SIZE=10
DEFAULT_LANGUAGE=en
```
