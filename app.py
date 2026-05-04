import os
import sys
from pathlib import Path

# ── Kill ChromaDB telemetry + OpenTelemetry before ANY chromadb import ──────
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# Monkey-patch otel_init to a no-op so chromadb never starts the gRPC exporter
# that causes the health/host-config connection errors on Windows
try:
    import chromadb.telemetry.opentelemetry as _chroma_otel
    _chroma_otel.otel_init = lambda *a, **kw: None
except Exception:
    pass

# ────────────────────────────────────────────────────────────────────────────

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import DEFAULT_LANGUAGE, MAX_RETRIEVED_CHUNKS
from ui.language_toggle import (
    render_language_selector_startup,
    render_language_toggle_topright,
    get_ui_text
)

st.set_page_config(
    page_title="Passport Assistant | पासपोर्ट सहायक",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .user-bubble {
        background: #0066CC; color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0 8px 20%; text-align: right;
    }
    .bot-bubble {
        background: #F0F2F6; color: #1a1a1a;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 20% 8px 0;
    }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "language_selected" not in st.session_state:
    st.session_state["language_selected"] = False
if "language" not in st.session_state:
    st.session_state["language"] = DEFAULT_LANGUAGE
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "memory" not in st.session_state:
    from memory.session_memory import SessionMemory
    st.session_state["memory"] = SessionMemory()
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# ── Language gate ─────────────────────────────────────────────────────────────
if not st.session_state["language_selected"]:
    render_language_selector_startup()  # calls st.stop() internally


# ── RAG pipeline (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base... this may take a minute on first run.")
def load_rag_pipeline():
    from rag.loader import DocumentLoader
    from rag.chunker import SmartChunker
    from rag.embedder import EmbeddingEngine
    from rag.vectorstore import PassportVectorStore
    from rag.retriever import HybridRetriever

    loader = DocumentLoader("data/raw", "passport_docs")
    chunker = SmartChunker()
    embedder = EmbeddingEngine()
    docs = loader.load_all()
    chunks = chunker.chunk_documents(docs)
    vectorstore = PassportVectorStore(embedder)
    vectorstore.add_chunks(chunks)
    retriever = HybridRetriever(vectorstore, chunks)
    return retriever, vectorstore


retriever, vectorstore = load_rag_pipeline()

from translation.translator import BilingualTranslator
from agents.router import IntentRouter
from rag.generator import ResponseGenerator

translator = BilingualTranslator()
router = IntentRouter()

# ── UI ────────────────────────────────────────────────────────────────────────
render_language_toggle_topright()

lang = st.session_state["language"]
generator = ResponseGenerator(language=lang)

st.title(get_ui_text("title"))
st.caption(get_ui_text("subtitle"))

with st.sidebar:
    st.header("ℹ️ Info")
    stats = vectorstore.get_collection_stats()
    st.metric("Knowledge Chunks", stats["total_chunks"])
    st.metric("Language", "Hindi 🇮🇳" if lang == "hi" else "English 🇬🇧")

    if st.button(get_ui_text("clear_button")):
        st.session_state["chat_history"] = []
        st.session_state["memory"].clear()
        st.session_state["last_query"] = ""
        st.rerun()

    st.divider()
    st.markdown("**Quick Topics:**")
    for topic in ["Fresh Passport", "Tatkal", "Renewal", "Documents", "Fees"]:
        if st.button(topic, use_container_width=True, key=f"quick_{topic}"):
            st.session_state["quick_query"] = topic
            st.rerun()

if not st.session_state["chat_history"]:
    st.info(get_ui_text("welcome"))

for role, text in st.session_state["chat_history"]:
    if role == "user":
        st.markdown(f'<div class="user-bubble">👤 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🛂 {text}</div>', unsafe_allow_html=True)

with st.expander(get_ui_text("upload_label")):
    uploaded_file = st.file_uploader(
        "Upload Aadhaar, Birth Certificate, etc.",
        type=["pdf", "png", "jpg", "jpeg"],
        key="doc_upload"
    )
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")

st.divider()

prefill = st.session_state.pop("quick_query", "")
user_input = prefill if prefill else st.chat_input(get_ui_text("input_placeholder"))

if user_input and user_input.strip():
    query = user_input.strip()

    if query == st.session_state["last_query"]:
        st.stop()

    st.session_state["last_query"] = query
    st.session_state["chat_history"].append(("user", query))

    with st.spinner(get_ui_text("thinking")):
        english_query = translator.translate_query_for_retrieval(query, lang)
        intent = router.classify(english_query)
        chunks = retriever.retrieve(english_query, n_results=MAX_RETRIEVED_CHUNKS)
        history = st.session_state["memory"].get_history()
        result = generator.generate(english_query, chunks, history)
        answer = result["answer"]

        if lang == "hi" and translator.detect_language(answer) == "en":
            answer = translator.to_hindi(answer)

        st.session_state["memory"].add_turn(query, answer, lang, intent)

        display_answer = answer
        if result.get("sources"):
            display_answer += f"\n\n*📚 Sources: {', '.join(result['sources'][:2])}*"

        st.session_state["chat_history"].append(("assistant", display_answer))

    st.rerun()
