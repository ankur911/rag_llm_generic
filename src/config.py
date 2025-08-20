
import os
from dotenv import load_dotenv
load_dotenv()

"""
General configuration and constants for the RAG pipeline.
Use of summerizer is optional, if used uncomment Summarizer (optional) to SUMMARIZER_MIN_LEN and
also modify llm.py, no effect on pipeline.py
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

USE_INSTRUCT_E5 = False   # set True to use the instruct-style implementation
EMBEDDING_MODEL = "intfloat/multilingual-e5-base" # E5, GTE, and BGE were specifically trained for asymmetric
# semantic search exact task a retriever performs in a RAG pipeline

# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # multilingual model,
# is a symmetric model, meaning it was trained to see if two sentences have the
# same meaning (paraphrasing). While this works for retrieval, it's not as 

# optimized as the newer, purpose-built models.
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # English model

DOCUMENT_SOURCES = [
    "https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding"
]
CHUNK_SIZE = 400  # lowering the chunksie from 1000 to 400
CHUNK_OVERLAP = 80 # lowering overlap 150 to 80
TOP_K_RESULTS = 10

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
HUGGINGFACEHUB_API_TOKEN: Set this env var to use HuggingFace Inference API.
If not set (or force_local=True), the pipeline uses the local model below.

LOCAL_LLM_ID: Local instruction-following model for RAG QA.
- Default (multilingual, instruction-tuned): bigscience/mt0-base  ✅
- Alternatives (commented):
    * English instruction models:
        - google/flan-t5-base
        - google/flan-t5-large
    * Summarizer (use via a router only, not as main QA model):
        - google/pegasus-xsum  (task='summarization')
    * Alternative Option: PaliGemma (multimodal VLM for text+images; not drop-in):
        - google/paligemma-3b-mix-224
        - Requires a vision-language pipeline; only consider if your RAG uses images.
"""
# LLM_REPO_ID = "google/gemma-2b-it"
# LLM_TASK = "text-generation"
# LLM_REPO_ID = "bigscience/mt0-base"      # or "google/flan-t5-base"
# LLM_TASK = "text2text-generation"
# LLM_REPO_ID = "google/flan-t5-small"   # try small/base
# LLM_TASK = "text2text-generation"
# LLM_REPO_ID = "google/flan-t5-small"
# LLM_TASK = "text2text-generation" 
LLM_REPO_ID = "bigscience/mt0-base"
LLM_TASK= "text2text-generation"

# Control where the LLM runs
USE_REMOTE_LLM = False  # set True to use Hugging Face (requires token)
HF_ENDPOINT_URL = None # for local and remote with serverless and 
#dedicated endpoint like  "https://<your-endpoint>.endpoints.huggingface.cloud"  
# for remote with dedicated end point

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "512"))

# Local QA model (multilingual, instruction-tuned)
LOCAL_LLM_ID = os.getenv("LOCAL_LLM_ID", "bigscience/mt0-base")
LOCAL_LLM_TASK = os.getenv("LOCAL_LLM_TASK", "text2text-generation")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "256"))


# Summarizer (optional)
# SUMMARIZER_MODEL_ID = os.getenv("SUMMARIZER_MODEL_ID", "google/pegasus-xsum")  # or "facebook/bart-large-cnn"
# SUMMARIZER_TASK = "summarization"
# # One-line answers, keep it tight
# SUMMARIZER_MAX_LEN = int(os.getenv("SUMMARIZER_MAX_LEN", "45"))   # ~45 tokens ≈ ~120–160 chars
# SUMMARIZER_MIN_LEN = int(os.getenv("SUMMARIZER_MIN_LEN", "18"))   # avoid 1–2 word outputs

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
#     f"You are a concise, factual assistant. Answer ONLY using the context.\n"
#     f"If the answer is not in the context, reply exactly: Not in knowledge base.\n"
#     f"Your ENTIRE answer must be <= {MAX_CHAR_LEN_RESP} characters.\n\n"
#     "Context:\n{context}\n\nQuestion:\n{question}\n\n"
#     f"Answer (<={MAX_CHAR_LEN_RESP} chars):"
# )

#Problem
# For Prompt above the response was:
# Query: List key WHO recommendations for infant and young child feeding.
# Answer: --- - provision of supportive health services with infant and young child feeding counselling during all contacts with caregivers and…
# Sources: ['https://www.who.int/news-room/fact-sheets/detail/infant-and-young-child-feeding'

# last implementation -> extend the template: Tighten the prompt (ban lists/markdown)
# Solution ->discovered that the model was mentioning strategies, programs, training, or 'WHO' so controlling that
PROMPT_TEMPLATE = (
    f"You are a concise, factual assistant. Answer ONLY using the context.\n"
    f"If the answer is not in the context, reply exactly: Not in knowledge base.\n"
    "Reply as ONE plain sentence of concrete recommendations; separate items with semicolons.\n"
    "Do NOT mention strategies, programmes, training, or 'WHO'. No bullets or markdown.\n"
    f"Your ENTIRE answer must be <= {MAX_CHAR_LEN_RESP} characters.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\n"
    f"Answer (<={MAX_CHAR_LEN_RESP} chars):"
)

PER_DOC_CHARS_ALLOWED = 900
OVERALL_CHARS_ALLOWED = 1600 # lowering from 2200 to 1600 for better context management

"""
=======Context compression (query-focused extraction)======
query-focused compression layer between retrieval and answer generation.
Instead of hand-curating phrases, we: (1) retrieve a broad set of 
chunks, (2) ask the model to extract one highly relevant sentence per 
chunk, and (3) stitch those compact “evidence” lines into the final 
context for the answer LLM. This works for any number of docs and keeps
the context short and on-target.
"""

# --- Context compression / ranking ---
USE_CONTEXT_COMPRESSION = True
COMPRESSION_MAX_SENTENCE_CHARS = 200   # cap each evidence line
COMPRESSION_TOPK = 8                   # keep top-N evidence lines

# --- Retrieval (MMR) ---
RETRIEVAL_FETCH_K = 80                 # candidate pool size for MMR
RETRIEVAL_TOP_K = TOP_K_RESULTS        # final k (you already control this)
RETRIEVAL_LAMBDA_MULT = 0.4            # 0=more diversity, 1=more relevance