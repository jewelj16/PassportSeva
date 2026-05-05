import re
from typing import Optional
from loguru import logger

INTENT_PATTERNS = {
    "eligibility": [
        r"eligible", r"can i apply", r"qualify", r"age limit",
        r"nationality", r"who can", r"पात्र", r"आवेदन कर सकते"
    ],
    "documents": [
        r"document", r"papers", r"proof", r"certificate", r"aadhaar",
        r"birth cert", r"what do i need", r"दस्तावेज़", r"कागज़"
    ],
    "fees": [
        r"fee", r"cost", r"price", r"charge", r"how much", r"amount",
        r"payment", r"शुल्क", r"कितना", r"पैसे"
    ],
    "tatkal": [
        r"tatkal", r"urgent", r"emergency", r"fast", r"quick",
        r"तत्काल", r"जल्दी"
    ],
    "appointment": [
        r"appointment", r"book", r"slot", r"psk", r"passport seva",
        r"visit", r"नियुक्ति", r"बुकिंग"
    ],
    "tracking": [
        r"status", r"track", r"where is", r"dispatch", r"delivered",
        r"स्थिति", r"ट्रैक"
    ],
    "reissue": [
        r"renew", r"reissue", r"expired", r"lost", r"damage",
        r"नवीनीकरण", r"खो गया"
    ],
    "minor": [
        r"child", r"minor", r"kid", r"son", r"daughter", r"baby",
        r"बच्चे", r"नाबालिग"
    ],
    "nri": [
        r"nri", r"abroad", r"foreign", r"overseas", r"outside india",
        r"प्रवासी"
    ],
    "photo": [
        r"photo", r"photograph", r"picture", r"specification",
        r"फोटो"
    ],
    "general": []
}


class IntentRouter:
    def classify(self, query: str) -> str:
        query_lower = query.lower()
        scores = {}
        for intent, patterns in INTENT_PATTERNS.items():
            if intent == "general":
                continue
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general"
        best_intent = max(scores, key=lambda k: scores[k])
        logger.debug(f"Intent '{best_intent}' for: {query[:50]}")
        return best_intent

    def get_category_filter(self, intent: str) -> Optional[str]:
        # Only filter for categories that actually exist in the data
        # "general" returns None so ChromaDB searches all chunks
        mapping = {
            "eligibility": "eligibility",
            "documents": "documents",
            "fees": "fees",
            "tatkal": "tatkal",
            "appointment": "process",
            "tracking": "general",    # broad — don't restrict
            "reissue": "general",
            "minor": "general",
            "nri": "nri",
            "photo": "documents",
            "general": None
        }
        return mapping.get(intent, None)
