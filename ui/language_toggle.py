import streamlit as st

UI_TEXT = {
    "en": {
        "title": "Passport Seva Assistant",
        "subtitle": "Your guide to Indian passport services",
        "welcome": "Hello! I'm your Passport Seva Assistant. Ask me anything about passport applications, documents, fees, or procedures.",
        "input_placeholder": "Ask anything about passports...",
        "send_button": "Send",
        "clear_button": "🗑️ Clear Chat",
        "lang_toggle_label": "🌐 Language",
        "sources_label": "📚 Sources",
        "confidence_label": "Confidence",
        "language_select": "Select Language / भाषा चुनें",
        "thinking": "Thinking...",
        "upload_label": "📎 Upload Document (optional)",
    },
    "hi": {
        "title": "पासपोर्ट सेवा सहायक",
        "subtitle": "भारतीय पासपोर्ट सेवाओं के लिए आपका मार्गदर्शक",
        "welcome": "नमस्ते! मैं आपका पासपोर्ट सेवा सहायक हूं। पासपोर्ट आवेदन, दस्तावेज़, शुल्क या प्रक्रियाओं के बारे में कुछ भी पूछें।",
        "input_placeholder": "पासपोर्ट के बारे में कुछ भी पूछें...",
        "send_button": "भेजें",
        "clear_button": "🗑️ चैट साफ़ करें",
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
    """Full-page language selector shown ONCE at startup — premium design."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .lang-page {
            display: flex; flex-direction: column; align-items: center;
            justify-content: center; padding: 80px 0 40px 0;
            font-family: 'Inter', sans-serif;
        }
        .lang-flag {
            font-size: 72px;
            margin-bottom: 16px;
        }
        .lang-emblem {
            width: 72px; height: 72px;
            background: #1e3a5f; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            color: white; font-size: 32px; font-weight: 700;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(30,58,95,0.25);
        }
        .lang-title {
            font-size: 28px; font-weight: 700; color: #111827;
            margin-bottom: 4px;
        }
        .lang-subtitle {
            font-size: 16px; color: #6B7280;
            margin-bottom: 32px;
        }
    </style>
    <div class="lang-page">
        <div class="lang-flag">🇮🇳</div>
        <div class="lang-emblem">🏛</div>
        <div class="lang-title">Passport Seva Assistant</div>
        <div class="lang-title" style="font-size:24px; margin-bottom:8px;">पासपोर्ट सेवा सहायक</div>
        <div class="lang-subtitle">
            Please select your preferred language &nbsp;|&nbsp; कृपया अपनी पसंदीदा भाषा चुनें
        </div>
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
