import os
import sys
from pathlib import Path
from datetime import datetime

# ── Kill ChromaDB telemetry + OpenTelemetry before ANY chromadb import ──────
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"
os.environ["OTEL_METRICS_EXPORTER"] = "none"
os.environ["OTEL_LOGS_EXPORTER"] = "none"

# Monkey-patch otel_init to a no-op so chromadb never starts the gRPC exporter
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
    get_ui_text
)

st.set_page_config(
    page_title="Passport Seva Assistant",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Master CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* ── Hide Streamlit defaults ────────────────────────────── */
    #MainMenu, footer, header { visibility: hidden !important; }
    .stDeployButton { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    /* Collapse default top padding */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 80px !important;
        max-width: 820px !important;
    }

    /* ── Custom Header Bar ──────────────────────────────────── */
    .psa-header {
        position: fixed; top: 0; left: 0; right: 0;
        z-index: 9999;
        background: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        padding: 12px 32px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .psa-header-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .psa-header-left .emblem {
        width: 36px; height: 36px;
        background: #1e3a5f;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: white; font-size: 18px; font-weight: 700;
    }
    .psa-header-left .title-text {
        font-size: 17px; font-weight: 600; color: #111827;
    }
    .psa-header-right {
        display: flex; align-items: center; gap: 8px;
        font-size: 14px; color: #374151;
    }

    /* ── Toggle Switch ──────────────────────────────────────── */
    .toggle-switch { position: relative; width: 44px; height: 24px; display: inline-block; }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .toggle-slider {
        position: absolute; cursor: pointer;
        top: 0; left: 0; right: 0; bottom: 0;
        background: #D1D5DB; border-radius: 24px;
        transition: 0.3s;
    }
    .toggle-slider:before {
        content: ""; position: absolute;
        height: 18px; width: 18px; left: 3px; bottom: 3px;
        background: white; border-radius: 50%;
        transition: 0.3s;
    }
    .toggle-switch input:checked + .toggle-slider { background: #2563EB; }
    .toggle-switch input:checked + .toggle-slider:before { transform: translateX(20px); }

    /* ── Spacer for fixed header ────────────────────────────── */
    .header-spacer { height: 70px; }

    /* ── Chat Messages ──────────────────────────────────────── */
    .chat-area {
        padding: 16px 0;
        display: flex; flex-direction: column; gap: 24px;
    }

    /* ── User Bubble ────────────────────────────────────────── */
    .user-msg-row {
        display: flex;
        justify-content: flex-end;
    }
    .user-msg-wrap { max-width: 65%; }
    .user-bubble {
        background: #2563EB;
        color: #FFFFFF;
        padding: 12px 20px;
        border-radius: 20px 20px 4px 20px;
        font-size: 15px;
        line-height: 1.5;
        word-wrap: break-word;
    }
    .user-ts {
        text-align: right;
        font-size: 11px; color: #9CA3AF;
        margin-top: 4px;
        padding-right: 4px;
    }
    .user-ts .checks { color: #2563EB; margin-left: 4px; }

    /* ── Bot Bubble ─────────────────────────────────────────── */
    .bot-msg-row {
        display: flex;
        justify-content: flex-start;
        align-items: flex-start;
        gap: 12px;
    }
    .bot-avatar {
        width: 40px; height: 40px; min-width: 40px;
        background: #1e3a5f;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 16px;
        margin-top: 2px;
    }
    .bot-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px 16px 16px 4px;
        padding: 18px 22px;
        max-width: 65%;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        font-size: 15px;
        line-height: 1.7;
        color: #1F2937;
        word-wrap: break-word;
    }
    .bot-card p { margin: 0 0 8px 0; }
    .bot-card ul, .bot-card ol { margin: 8px 0; padding-left: 20px; }
    .bot-card li { margin-bottom: 4px; }

    /* ── Source Citation ─────────────────────────────────────── */
    .source-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 6px;
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #F3F4F6;
        font-size: 13px;
        color: #6B7280;
    }
    .source-icon {
        color: #2563EB; font-size: 14px;
    }
    .source-name {
        color: #2563EB;
        font-weight: 500;
        text-decoration: none;
        word-break: break-all;
    }

    .bot-ts {
        font-size: 11px; color: #9CA3AF;
        margin-top: 4px;
        padding-left: 52px;
    }

    /* ── Bottom Disclaimer ──────────────────────────────────── */
    .disclaimer {
        text-align: center;
        font-size: 12px;
        color: #9CA3AF;
        padding: 6px 0 0 0;
    }

    /* ── Welcome Card ───────────────────────────────────────── */
    .welcome-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        max-width: 520px;
        margin: 40px auto;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .welcome-card .emblem-lg {
        width: 56px; height: 56px;
        background: #1e3a5f;
        border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        color: white; font-size: 24px; font-weight: 700;
        margin-bottom: 16px;
    }
    .welcome-card h3 {
        color: #111827; margin: 0 0 8px 0; font-size: 20px;
    }
    .welcome-card p {
        color: #6B7280; font-size: 15px; margin: 0;
        line-height: 1.6;
    }

    /* ── Sidebar overrides ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }

    /* ── Streamlit chat_input styling ───────────────────────── */
    div[data-testid="stChatInput"] textarea {
        border-radius: 24px !important;
        border: 1px solid #D1D5DB !important;
        padding: 12px 20px !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
    }
    div[data-testid="stChatInput"] textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    div[data-testid="stChatInput"] button {
        background: #2563EB !important;
        color: white !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
    }
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
    render_language_selector_startup()


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

lang = st.session_state["language"]
generator = ResponseGenerator(language=lang)

# ── Custom Header ─────────────────────────────────────────────────────────────
toggle_label = "Switch to Hindi" if lang == "en" else "अंग्रेज़ी में बदलें"

st.markdown(f"""
<div class="psa-header">
    <div class="psa-header-left">
        <div class="emblem">🏛</div>
        <span class="title-text">Passport Seva Assistant</span>
        <span style="font-size:22px;">🇮🇳</span>
    </div>
    <div class="psa-header-right">
        <span style="font-size:14px;">{toggle_label}</span>
    </div>
</div>
<div class="header-spacer"></div>
""", unsafe_allow_html=True)

# Hidden but functional Streamlit toggle for Hindi (placed unobtrusively)
col_spacer, col_toggle = st.columns([8, 1])
with col_toggle:
    hindi_on = st.toggle("हिंदी", value=(lang == "hi"), label_visibility="collapsed")
    if hindi_on and lang == "en":
        st.session_state["language"] = "hi"
        st.rerun()
    elif not hindi_on and lang == "hi":
        st.session_state["language"] = "en"
        st.rerun()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ℹ️ Info")
    stats = vectorstore.get_collection_stats()
    st.metric("Knowledge Chunks", stats["total_chunks"])
    st.metric("Language", "Hindi 🇮🇳" if lang == "hi" else "English 🇬🇧")

    if st.button(get_ui_text("clear_button"), use_container_width=True):
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


# ── Helper: format timestamp ──────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime("%-I:%M %p")


# ── Helper: render a single message ──────────────────────────────────────────
def render_message(role, text, timestamp=""):
    ts = timestamp or _ts()
    if role == "user":
        st.markdown(f"""
        <div class="user-msg-row">
            <div class="user-msg-wrap">
                <div class="user-bubble">{text}</div>
                <div class="user-ts">{ts}<span class="checks"> ✓✓</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # --- Unpack sources from tuple or extract from text ---
        if isinstance(text, tuple):
            main_text, sources_list = text
        else:
            main_text = text
            sources_list = []

        # Strip any LLM-generated [Source: ...] lines from the answer
        import re as _re
        main_text = _re.sub(r'\[Source:.*?\]', '', main_text).strip()
        # Also strip our appended *📚 Sources: ...* if present
        if '📚 Sources:' in main_text:
            main_text = main_text.split('📚 Sources:')[0].strip().rstrip('*').strip()

        # Build source citation HTML
        source_html = ""
        if sources_list:
            source_links = "".join(
                f'<span class="source-name">{s}</span> ' for s in sources_list
            )
            source_html = f'<div class="source-row"><span class="source-icon">📄</span><span style="color:#6B7280;">Source:</span>{source_links}</div>'

        # Convert newlines to HTML breaks
        display_text = main_text.replace("\n", "<br>")

        final_html = f'<div class="bot-msg-row"><div class="bot-avatar">🏛</div><div class="bot-card">{display_text}{source_html}</div></div><div class="bot-ts">{ts}</div>'
        st.markdown(final_html, unsafe_allow_html=True)


# ── Chat Area ─────────────────────────────────────────────────────────────────
if not st.session_state["chat_history"]:
    welcome_msg = get_ui_text("welcome")
    st.markdown(f"""
    <div class="welcome-card">
        <div class="emblem-lg">🏛</div>
        <h3>{get_ui_text("title").replace("🛂 ", "")}</h3>
        <p>{welcome_msg}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    for role, text in st.session_state["chat_history"]:
        render_message(role, text)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Chat Input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("quick_query", "")
user_input = prefill if prefill else st.chat_input(get_ui_text("input_placeholder"))

st.markdown(
    '<div class="disclaimer">Passport Seva Assistant can make mistakes. '
    'Please verify important information on the official Passport Seva Portal.</div>',
    unsafe_allow_html=True
)

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

        # Store answer + sources as a tuple so render_message can handle them cleanly
        sources = result.get("sources", [])[:3]
        st.session_state["chat_history"].append(("assistant", (answer, sources)))

    st.rerun()
