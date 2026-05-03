import streamlit as st
from ui.components import render_chat_bubble


def render_chat_history(chat_history: list):
    """Render the full chat history."""
    for role, text in chat_history:
        render_chat_bubble(role, text)
