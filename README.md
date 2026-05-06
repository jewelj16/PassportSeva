# Passport Seva Assistant

> A RAG-based bilingual chatbot for Indian Passport services

[![Live Demo](https://img.shields.io/badge/Live%20Demo-passportseva.streamlit.app-brightgreen)](https://passportseva.streamlit.app)
[![Demo Video](https://img.shields.io/badge/Demo%20Video-YouTube-red?logo=youtube)](https://www.youtube.com/watch?v=mecmjJPnTHI)

## Overview

Navigating Indian passport documentation can be tedious. **Passport Seva Assistant** retrieves relevant passages from official government PDFs and generates grounded, cited answers via a Streamlit chat interface. It supports bilingual (English/Hindi) interaction.

## Architecture

| Component | Technology |
|---|---|
| **Retrieval** | Hybrid BM25 + Dense vector search (ChromaDB + MiniLM-L6-v2) |
| **Generation** | Groq — Llama-3.3-70B |
| **Memory** | Sliding window session memory (deque, size 10) |
| **Translation** | Google Translate API (Hindi ↔ English) |
| **Routing** | Intent-based router to specialized sub-agents |
| **UI** | Streamlit with custom chat interface |

## Key Features

- 🔍 **Hybrid Retrieval:** Combines dense cosine similarity (ChromaDB) with BM25 sparse matching (α=0.7) for highly accurate context retrieval across 1,550 chunks of official government documents.
- 📄 **Source Citations:** Every response includes page-level citations pointing to the exact source document, preventing hallucinations.
- 🧠 **Session Memory:** Sliding window retains last 10 conversation turns so follow-up questions work naturally.
- 🌐 **Bilingual Support:** Automatically detects Hindi input, translates for the retrieval pipeline, and returns the response in Hindi.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/jewelj16/PassportSeva.git
   cd PassportSeva
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up your `.env` file (see [Environment Variables](#environment-variables) below).
5. Build the knowledge base (one-time step):
   ```bash
   python scripts/build_kb.py
   ```
6. Launch the app:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
PassportSeva/
├── app.py                    # Streamlit entry point
├── config/settings.py        # Configuration & system prompts
├── data/
│   ├── raw/                  # 7 TXT docs + 8 PDFs + curated Q&A JSONL
│   ├── processed/            # Chunked data
│   └── evaluation/           # Test sets (EN + HI)
├── rag/                      # Pipeline: loader, chunker, embedder, vectorstore, retriever, generator
├── agents/                   # Intent router
├── memory/                   # Sliding window session memory
├── translation/              # Hindi ↔ English translation
├── ui/                       # Streamlit UI components
├── scripts/                  # build_kb.py & run_eval.py
└── report/                   # LaTeX project report
```

## Environment Variables

Use a **Groq API key** (free tier available) for the LLM:

```env
GROQ_API_KEY=<your_groq_api_key>
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PERSIST_PATH=./chroma_db
COLLECTION_NAME=passport_knowledge_base
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MAX_RETRIEVED_CHUNKS=5
MEMORY_WINDOW_SIZE=10
DEFAULT_LANGUAGE=en
```
