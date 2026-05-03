# Assignment 3 — Topic T10.1: Passport Assistant
## Statistical Methods in AI (SMAI) — Project Reference Document

---

## Assignment Overview

**Assignment:** A3 — Build a Domain-Specific RAG-Based Conversational AI System  
**Topic Number:** T10.1  
**Topic Name:** Passport Assistant  
**Type:** Group Project

---

## Topic Details (T10.1 — Passport Assistant)

### Core Description
Build a conversational AI assistant specifically for passport-related queries in India. The system must help users navigate the passport application process, understand documentation requirements, check eligibility, and get answers to common passport-related questions.

---

## Required Deliverables (from Assignment PDF)

### 1. RAG Pipeline
- Implement a **Retrieval-Augmented Generation (RAG)** pipeline for the domain
- Must include a **vector database** for storing and retrieving relevant document chunks
- The pipeline must ground LLM responses in retrieved factual context

### 2. Conversational Interface
- Build a working **chat-based UI** that users can interact with
- Must maintain **conversation history / memory** across turns within a session
- Users should be able to ask follow-up questions naturally

### 3. Knowledge Base
- Curate and preprocess a **domain-specific knowledge base** (passport rules, MEA guidelines, document requirements, Tatkal process, etc.)
- Documents must be properly **chunked, embedded, and indexed**

### 4. Multimodal Input (where applicable)
- Support for **text input** at minimum
- Bonus/extension: Accept **image/document uploads** (e.g., scanning an Aadhaar or checking a passport photo)

### 5. Evaluation
- Evaluate the system on **relevance, faithfulness, and answer correctness**
- Report metrics such as retrieval accuracy, response quality, and latency

### 6. Report / Documentation
- Submit a written report describing:
  - System architecture
  - Data collection and preprocessing
  - Embedding and retrieval strategy
  - LLM integration approach
  - Evaluation methodology and results
  - Challenges and future work

---

## Domain Scope: What the Passport Assistant Must Cover

Based on topic T10.1, the assistant must be knowledgeable about:

| Category | Specific Topics |
|---|---|
| Application Types | Fresh passport, Reissue, Tatkal, Minor passport, Diplomatic |
| Documents Required | Aadhaar, birth certificate, address proof, photo ID, existing passport |
| Eligibility Rules | Age, nationality, existing travel document status |
| Passport Seva Kendra | PSK location, appointment booking, walk-in rules |
| Fees | Normal fee, Tatkal fee, minor fee, booklet size fees |
| Processing Time | Normal (30–45 days), Tatkal (1–3 days) |
| Online Portal | passportindia.gov.in, DigiLocker integration |
| Police Verification | When required, how long it takes, types |
| Renewal Process | Expiry rules, damage/lost passport reissue |
| Annexures | Annexure A–H, when each is required |
| NRI / OCI | Special rules for NRI applicants |
| Name Change / Correction | Marriage, legal name change process |
| Photo Guidelines | Size, background, specifications |
| Grievance Redressal | Helpline numbers, escalation process |

---

## Technical Requirements (Extracted from Assignment)

### Must-Have
- [ ] Working RAG pipeline (retrieval + generation)
- [ ] Vector store with domain documents indexed
- [ ] LLM integration for response generation
- [ ] Chat UI with session memory
- [ ] Chunking and embedding pipeline documented
- [ ] At least 50–100 domain-specific Q&A pairs for evaluation

### Should-Have
- [ ] Bilingual support (Hindi + English)
- [ ] Source citation in responses ("According to MEA guidelines...")
- [ ] Confidence score or "I don't know" fallback
- [ ] Response grounded only in retrieved context (no hallucination)

### Good-to-Have
- [ ] Multimodal input (photo/document upload)
- [ ] Agentic routing (eligibility check agent, document agent, etc.)
- [ ] Deployment (Streamlit Cloud / Hugging Face Spaces)

---

## Grading Criteria (as inferred from assignment structure)

| Component | Weight (Estimated) |
|---|---|
| RAG Pipeline correctness and quality | High |
| Knowledge base coverage and preprocessing | High |
| Conversational UI and UX | Medium |
| Evaluation methodology and results | High |
| Report quality and documentation | Medium |
| Bonus: Multimodal / Bilingual / Agentic | Bonus |

---

## Tech Stack Chosen for T10.1

| Layer | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (via AI Studio — free) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (CPU, free) |
| Vector DB | ChromaDB (local, persistent) |
| Translation | Google Translate API (googletrans library) |
| UI | Streamlit |
| Memory | LangChain ConversationBufferWindowMemory |
| Orchestration | LangChain |
| Knowledge Base | MEA passport seva docs, scraped FAQs, custom Q&A pairs |

---

## Notes
- All domain knowledge must be sourced from official GOI / MEA / Passport Seva portal
- Responses must cite source chunks to prevent hallucination
- Hindi interface must be functionally equivalent to English interface
- Evaluation must be done on a held-out test set, not the training/indexing data
