import streamlit as st


def render_chat_bubble(role: str, text: str):
    """Render a single chat bubble."""
    if role == "user":
        st.markdown(f'<div class="user-bubble">👤 {text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">🛂 {text}</div>', unsafe_allow_html=True)


def render_confidence_badge(confidence: str):
    """Render a confidence indicator badge."""
    colors = {"high": "#28a745", "medium": "#ffc107", "low": "#dc3545"}
    labels = {"high": "High Confidence", "medium": "Medium Confidence", "low": "Low Confidence"}
    color = colors.get(confidence, "#888")
    label = labels.get(confidence, confidence)
    st.markdown(
        f'<span style="color:{color}; font-size:0.8em;">● {label}</span>',
        unsafe_allow_html=True
    )


def render_source_tags(sources: list[str]):
    """Render source citation tags."""
    if sources:
        tags = " • ".join(sources[:3])
        st.markdown(
            f'<div class="source-tag">📚 Sources: {tags}</div>',
            unsafe_allow_html=True
        )
