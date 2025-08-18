
import os
from dotenv import load_dotenv
load_dotenv()

"""
General configuration and constants for the RAG pipeline.
"""

# === Vector Store & Embedding ===
"""
VECTOR_STORE_PATH: Path to the FAISS vector store.
EMBEDDING_MODEL: HuggingFace model for embeddings.
DOCUMENT_SOURCES: List of URLs for document ingestion.
CHUNK_SIZE, CHUNK_OVERLAP: Document chunking parameters.
TOP_K_RESULTS: Number of top results to retrieve.
"""
VECTOR_STORE_PATH = "faiss_vector_store"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DOCUMENT_SOURCES = [
    "https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding"
]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K_RESULTS = 3

# === Query Classification ===
"""
TOXIC_KEYWORDS: Set of keywords for toxicity detection.
PROFANITY_WORDS: Set of words for profanity detection.
"""
TOXIC_KEYWORDS = {"kill", "hate", "suicide", "abuse", "stupid", "idiot", "die", "murder"}
PROFANITY_WORDS = {"damn", "shit", "fuck", "bitch", "bastard"}

# === LLM Configuration ===
"""
LLM_REPO_ID: HuggingFace repo ID for remote LLM.
LLM_TASK: Task for remote LLM (e.g., text-generation).
HUGGINGFACEHUB_API_TOKEN: Set this environment variable to use HuggingFace Inference API.
If not set, the pipeline will use the local fallback model.
LOCAL_LLM_ID: Local model ID for fallback.
LOCAL_LLM_TASK: Local pipeline task (e.g., text2text-generation).
LOCAL_LLM_MAX_NEW_TOKENS: Max new tokens for local generation.
"""

LLM_REPO_ID = "google/gemma-2b-it"
LLM_TASK = "text-generation"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

LOCAL_LLM_ID = os.getenv("LOCAL_LLM_ID", "google/flan-t5-base")
LOCAL_LLM_TASK = os.getenv("LOCAL_LLM_TASK", "text2text-generation")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))

# === Pipeline & Prompt Parameters ===
"""
MAX_CHAR_LEN_RESP: Max character length for generated answers.
KNOWLEDGE_BASE_NOT_LOADED: Message if vector store is not loaded.
NO_DOCS_FOUND: Message if no docs are retrieved.
PROMPT_TEMPLATE: Prompt template for RAG answer generation.
PER_DOC_CHARS_ALLOWED: Max chars per doc in context.
OVERALL_CHARS_ALLOWED: Max chars for overall context.
"""
MAX_CHAR_LEN_RESP = 140
KNOWLEDGE_BASE_NOT_LOADED = "Knowledge base not loaded."
NO_DOCS_FOUND = "I couldn't retrieve relevant context from the knowledge base."
# PROMPT_TEMPLATE = (
#     "You are a concise, factual assistant. Answer ONLY using the context.\n"
#     "Your ENTIRE answer must be <= {MAX_CHAR_LEN_RESP} characters.\n"
#     "If the answer is not in the context, say so briefly.\n\n"
#     "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (<=140 chars):"
# )
PROMPT_TEMPLATE = (
    f"You are a concise, factual assistant. Answer ONLY using the context.\n"
    f"Your ENTIRE answer must be <= {MAX_CHAR_LEN_RESP} characters.\n"
    "If the answer is not in the context, say so briefly.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
    f"Answer (<={MAX_CHAR_LEN_RESP} chars):"
)
PER_DOC_CHARS_ALLOWED = 900
OVERALL_CHARS_ALLOWED = 2200
