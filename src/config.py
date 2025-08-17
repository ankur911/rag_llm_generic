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
