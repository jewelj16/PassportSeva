import re
from loguru import logger


class BilingualTranslator:
    HINDI_CHARS = re.compile(r'[\u0900-\u097F]')

    def __init__(self):
        # Lazy-load translators to avoid import-time network errors
        self._en_to_hi = None
        self._hi_to_en = None

    def _get_en_to_hi(self):
        if self._en_to_hi is None:
            from deep_translator import GoogleTranslator
            self._en_to_hi = GoogleTranslator(source='en', target='hi')
        return self._en_to_hi

    def _get_hi_to_en(self):
        if self._hi_to_en is None:
            from deep_translator import GoogleTranslator
            self._hi_to_en = GoogleTranslator(source='hi', target='en')
        return self._hi_to_en

    def detect_language(self, text: str) -> str:
        hindi_count = len(self.HINDI_CHARS.findall(text))
        total_chars = len(text.replace(" ", ""))
        if total_chars == 0:
            return "en"
        return "hi" if (hindi_count / total_chars) > 0.15 else "en"

    def to_english(self, text: str) -> str:
        try:
            detected = self.detect_language(text)
            if detected == "hi":
                translated = self._get_hi_to_en().translate(text)
                logger.debug(f"HI→EN: '{text[:50]}' → '{translated[:50]}'")
                return translated
            return text
        except Exception as e:
            logger.warning(f"Translation HI→EN failed: {e}. Using original.")
            return text

    def to_hindi(self, text: str) -> str:
        try:
            return self._get_en_to_hi().translate(text)
        except Exception as e:
            logger.warning(f"Translation EN→HI failed: {e}. Returning English.")
            return text

    def translate_query_for_retrieval(self, query: str, ui_language: str) -> str:
        """
        Translate to English ONLY if the query itself is in Hindi.
        Do NOT translate if the user just switched UI to Hindi but is typing in English —
        that would corrupt the embedding lookup.
        """
        detected = self.detect_language(query)
        if detected == "hi":
            return self.to_english(query)
        return query  # Already English — return as-is regardless of ui_language
