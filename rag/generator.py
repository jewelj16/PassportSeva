from config.settings import GOOGLE_API_KEY, LLM_MODEL, SYSTEM_PROMPT_EN, SYSTEM_PROMPT_HI


class ResponseGenerator:
    def __init__(self, language: str = "en"):
        self.language = language
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            if not GOOGLE_API_KEY:
                raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")
            self._client = genai.Client(api_key=GOOGLE_API_KEY)
        return self._client

    def generate(self, query: str, retrieved_chunks: list, conversation_history: list = None) -> dict:
        system_prompt = SYSTEM_PROMPT_HI if self.language == "hi" else SYSTEM_PROMPT_EN
        context = self._build_context(retrieved_chunks)

        history_str = ""
        if conversation_history:
            for turn in conversation_history[-4:]:
                history_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        prompt = f"""{system_prompt}

CONVERSATION HISTORY:
{history_str}
RETRIEVED CONTEXT:
{context}

CURRENT QUESTION: {query}

Provide a clear, helpful answer based ONLY on the context above.
At the end, mention which source(s) you used in format: [Source: ...]
If you cannot answer from context, say so clearly.
"""
        try:
            from google import genai
            from google.genai import types

            client = self._get_client()
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=1024,
                    top_p=0.95,
                )
            )

            if not response.text:
                raise ValueError("Empty response (possibly blocked by safety filters).")

            sources = list({c["source"] for c in retrieved_chunks[:3]})
            return {
                "answer": response.text,
                "sources": sources,
                "confidence": self._assess_confidence(retrieved_chunks),
                "retrieved_count": len(retrieved_chunks)
            }
        except Exception as e:
            print(f"[ERROR] Gemini failed: {e}")
            fallback = (
                "मैं अभी उत्तर देने में असमर्थ हूं। कृपया passportindia.gov.in पर जाएं।"
                if self.language == "hi" else
                f"Unable to answer right now ({e}). Please visit passportindia.gov.in or call 1800-258-1800."
            )
            return {"answer": fallback, "sources": [], "confidence": "low"}

    def _build_context(self, chunks: list) -> str:
        if not chunks:
            return "No relevant context found."
        return "\n\n---\n\n".join(
            f"[{i}] (Source: {c['source']})\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )

    def _assess_confidence(self, chunks: list) -> str:
        if not chunks:
            return "low"
        avg = sum(c.get("score", 0) for c in chunks) / len(chunks)
        return "high" if avg > 0.75 else "medium" if avg > 0.5 else "low"
