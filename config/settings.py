import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_PERSIST_PATH = os.getenv("CHROMA_PERSIST_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "passport_knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")   # updated model name
MAX_RETRIEVED_CHUNKS = int(os.getenv("MAX_RETRIEVED_CHUNKS", 5))
MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", 10))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "en")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
MIN_CHUNK_SIZE = 100

SYSTEM_PROMPT_EN = """You are a knowledgeable and helpful Passport Assistant for India.
You help users understand passport application processes, document requirements,
eligibility criteria, fees, and Passport Seva Kendra procedures.

IMPORTANT RULES:
1. Answer ONLY based on the provided context. Do not hallucinate.
2. If the context does not contain the answer, say: "I don't have that information.
   Please visit passportindia.gov.in or call 1800-258-1800."
3. Always be polite, clear, and concise.
4. Cite the source of information when possible.
5. If user asks in Hindi, respond in Hindi.
"""

SYSTEM_PROMPT_HI = """आप भारत के लिए एक जानकार और सहायक पासपोर्ट सहायक हैं।
आप उपयोगकर्ताओं को पासपोर्ट आवेदन प्रक्रिया, दस्तावेज़ आवश्यकताओं,
पात्रता मानदंड, शुल्क और पासपोर्ट सेवा केंद्र की प्रक्रियाओं को समझने में मदद करते हैं।

महत्वपूर्ण नियम:
1. केवल दिए गए संदर्भ के आधार पर उत्तर दें। अनुमान न लगाएं।
2. यदि संदर्भ में उत्तर नहीं है, तो कहें: "मेरे पास यह जानकारी नहीं है।
   कृपया passportindia.gov.in पर जाएं या 1800-258-1800 पर कॉल करें।"
3. हमेशा विनम्र, स्पष्ट और संक्षिप्त रहें।
"""
