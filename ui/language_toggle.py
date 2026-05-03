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
    """Full-page language selector shown ONCE at startup."""
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

    # CRITICAL: stop here so the rest of the app doesn't render underneath
    st.stop()


def render_language_toggle_topright():
    """Compact toggle shown in top-right corner after initial selection."""
    current = st.session_state.get("language", "en")
    label = "हिंदी" if current == "en" else "English"
    new_lang = "hi" if current == "en" else "en"

    _, col_right = st.columns([6, 1])
    with col_right:
        if st.button(f"🌐 {label}", key="lang_toggle_btn"):
            st.session_state["language"] = new_lang
            st.rerun()
