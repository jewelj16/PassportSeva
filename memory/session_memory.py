from collections import deque
from typing import Optional, List, Dict


class Turn:
    __slots__ = ("user", "assistant", "language", "intent")

    def __init__(self, user: str, assistant: str, language: str = "en", intent: Optional[str] = None):
        self.user = user
        self.assistant = assistant
        self.language = language
        self.intent = intent


class SessionMemory:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.turns: deque = deque(maxlen=window_size)
        self.session_language: str = "en"
        self.user_profile: Dict = {}

    def add_turn(self, user_msg: str, assistant_msg: str, language: str = "en", intent: str = None):
        self.turns.append(Turn(
            user=user_msg,
            assistant=assistant_msg,
            language=language,
            intent=intent
        ))

    def get_history(self) -> List[Dict]:
        return [{"user": t.user, "assistant": t.assistant} for t in self.turns]

    def get_last_n(self, n: int = 3) -> List[Dict]:
        recent = list(self.turns)[-n:]
        return [{"user": t.user, "assistant": t.assistant} for t in recent]

    def get_context_summary(self) -> str:
        if not self.turns:
            return "This is the start of the conversation."
        recent_intents = [t.intent for t in self.turns if t.intent]
        topics = list(dict.fromkeys(recent_intents[-3:]))  # ordered unique
        return f"Recent topics: {', '.join(topics) if topics else 'general inquiry'}"

    def clear(self):
        self.turns.clear()
        self.user_profile.clear()

    def update_user_profile(self, key: str, value):
        self.user_profile[key] = value

    def get_user_profile(self) -> Dict:
        return self.user_profile
