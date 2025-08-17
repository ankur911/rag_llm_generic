import os
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
"""
Configuration and constants for RAG pipeline.
"""
DOCUMENT_SOURCES = [
    "https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding"
]
VECTOR_STORE_PATH = "faiss_vector_store"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_REPO_ID = "google/gemma-2b-it"
LLM_TASK = "text-generation"
TOP_K_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOXIC_KEYWORDS = {"kill", "hate", "suicide", "abuse", "stupid", "idiot", "die", "murder"}
PROFANITY_WORDS = {"damn", "shit", "fuck", "bitch", "bastard"}
LOCAL_LLM_ID = os.getenv("LOCAL_LLM_ID", "google/flan-t5-base")
LOCAL_LLM_TASK = os.getenv("LOCAL_LLM_TASK", "text2text-generation")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))
MAX_CHAR_LEN_RESP = 140
KNOWLEDGE_BASE_NOT_LOADED = "Knowledge base not loaded."
NO_DOCS_FOUND = "I couldn't retrieve relevant context from the knowledge base."
PROMPT_TEMPLATE = (
    "You are a concise, factual assistant. Answer ONLY using the context.\n"
    "Your ENTIRE answer must be <= 140 characters.\n"
    "If the answer is not in the context, say so briefly.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (<=140 chars):"
)
PER_DOC_CHARS_ALLOWED = 900
OVERALL_CHARS_ALLOWED = 2200
