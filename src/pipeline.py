"""
Pipeline nodes and orchestration for RAG pipeline.
Combines embed-ranked context compression with fallback to naive context building.
"""
import re
import math
from functools import lru_cache
from langchain_core.prompts import PromptTemplate

from src.llm import get_llm
from src.vectorstore import VectorStoreManager
from src.config import (
    TOP_K_RESULTS, TOXIC_KEYWORDS, PROFANITY_WORDS, SUMMARY_KEYWORDS,
    MAX_CHAR_LEN_RESP, SUMMARY_MAX_CHAR_LEN_RESP, KNOWLEDGE_BASE_NOT_LOADED, NO_DOCS_FOUND,
    PROMPT_TEMPLATE, SUMMARY_PROMPT_TEMPLATE, PER_DOC_CHARS_ALLOWED, OVERALL_CHARS_ALLOWED,
    USE_CONTEXT_COMPRESSION, COMPRESSION_MAX_SENTENCE_CHARS, COMPRESSION_TOPK,
    RETRIEVAL_FETCH_K, RETRIEVAL_LAMBDA_MULT,
)

# --- Text processing helpers ---

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
LIST_BULLET_RE = re.compile(r'^\s*(?:[-–—•*]\s+)+', re.M)
SEP_LINE_RE = re.compile(r'^\s*---\s*$', re.M)

def clean_text(s: str) -> str:
    """Remove redundant whitespace, list bullets, and separator lines."""
    s = (s or "").strip()
    s = SEP_LINE_RE.sub(" ", s)
    s = LIST_BULLET_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def enforce_char_limit(text: str, limit: int = MAX_CHAR_LEN_RESP) -> str:
    """Hard-cap text length, trimming at a word boundary."""
    s = " ".join((text or "").split()).strip()
    if len(s) <= limit:
        return s
    cut = s[: max(0, limit - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "…"

def enforce_min_len(ans: str, min_chars: int = 10) -> str:
    """Replace short/junk answers with a canonical refusal."""
    return ans if len((ans or "").strip()) >= min_chars else "Not in knowledge base."

def _dedup_keep_order(lines):
    """Deduplicate lines in a list while preserving order."""
    kept, seen = [], set()
    for s in lines:
        k = s.strip().lower()
        if k in seen:
            continue
        seen.add(k)
        kept.append(s.strip())
    return kept

# --- Query safety classification ---

def classify_query(text: str):
    """Flag queries with toxic, profane, or summary keywords."""
    lower = (text or "").lower()
    flags = set()
    if any(w in lower for w in TOXIC_KEYWORDS):
        flags.add("toxic")
    if any(w in lower for w in PROFANITY_WORDS):
        flags.add("profanity")
    
    # Recommendation 2: Check for summary keywords
    is_summary = any(w in lower for w in SUMMARY_KEYWORDS)
    
    return {"is_safe": not bool(flags), "flags": sorted(flags), "is_summary": is_summary}

# --- Context building ---

def _cosine(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def build_context_naive(docs, per_doc_chars=PER_DOC_CHARS_ALLOWED, overall_chars=OVERALL_CHARS_ALLOWED) -> str:
    """Naive context builder: concatenate doc chunks up to a character limit."""
    parts, budget = [], overall_chars
    for d in docs:
        text = (d.page_content or "")[:per_doc_chars]
        text = " ".join(text.split())
        if not text:
            continue
        add = text[:budget]
        parts.append(add)
        budget -= len(add) + 4  # Account for separator
        if budget <= 0:
            break
    return "\n---\n".join(parts).strip()

def build_context_embed_rank(
    manager,
    docs,
    query: str,
    per_doc_chars: int = PER_DOC_CHARS_ALLOWED,
    overall_chars: int = OVERALL_CHARS_ALLOWED,
    top_sentences: int = COMPRESSION_TOPK,
) -> str:
    """
    Embed-ranked context compression:
    1. Split docs into sentences.
    2. Filter out irrelevant or meta sentences.
    3. Rank sentences by cosine similarity to the query.
    4. Pack the top-N unique sentences into the context.
    """
    candidates = []
    for d in docs:
        chunk = (d.page_content or "")[:per_doc_chars]
        sentences = [s.strip() for s in SENT_SPLIT.split(chunk) if s.strip()]
        sentences = [clean_text(s) for s in sentences if len(s) >= 30]
        
        ban = ("strategy", "programme", "program", "training", "course", "who provides", "who and unicef", "monitoring", "report", "launch")
        sentences = [s for s in sentences if not any(b in s.lower() for b in ban)]
        sentences = [s[:COMPRESSION_MAX_SENTENCE_CHARS].rstrip(" ,;:") for s in sentences]
        candidates.extend(sentences)

    if not candidates:
        return ""

    qv = manager.embedding_model.embed_query(query)
    sv = manager.embedding_model.embed_documents(candidates)
    scored = sorted(zip(candidates, sv), key=lambda p: _cosine(p[1], qv), reverse=True)

    top = _dedup_keep_order([c for c, _ in scored][:max(3, top_sentences)])
    
    ctx, used = [], 0
    for s in top:
        if used + len(s) + 1 > overall_chars:
            break
        ctx.append(s)
        used += len(s) + 1
    return "\n".join(ctx).strip()

# --- Main RAG function ---

@lru_cache(maxsize=1)
def _qa_llm(is_summary: bool = False):
    """Cached LLM instance for reuse."""
    return get_llm(is_summary=is_summary)

def retrieve_and_generate(manager, llm, query: str, return_context: bool = False) -> dict:
    """
    Main RAG pipeline function:
    1. Load vector store if not already loaded.
    2. Retrieve documents using MMR.
    3. Build context (with optional embed-ranked compression).
    4. Generate an answer using the LLM.
    5. Apply post-processing to the answer.
    """
    if manager.vector_store is None and not manager.load():
        base = {"answer": KNOWLEDGE_BASE_NOT_LOADED, "sources": []}
        return {**base, "context": ""} if return_context else base

    retriever = manager.vector_store.as_retriever(
        search_kwargs={
            "k": TOP_K_RESULTS,
            "fetch_k": RETRIEVAL_FETCH_K,
            "mmr": True,
            "lambda_mult": RETRIEVAL_LAMBDA_MULT,
        }
    )
    docs = retriever.invoke(query)
    if not docs:
        base = {"answer": NO_DOCS_FOUND, "sources": []}
        return {**base, "context": ""} if return_context else base

    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})

    # Build context
    if USE_CONTEXT_COMPRESSION:
        context = build_context_embed_rank(manager, docs, query)
        if not context:  # Fallback if compression yields nothing
            context = build_context_naive(docs)
    else:
        context = build_context_naive(docs)

    # Recommendation 2: Choose prompt based on query type
    query_meta = classify_query(query)
    if query_meta["is_summary"]:
        prompt_template = SUMMARY_PROMPT_TEMPLATE
    else:
        prompt_template = PROMPT_TEMPLATE
    
    prompt = PromptTemplate.from_template(prompt_template)
    llm_used = llm or _qa_llm(is_summary=query_meta["is_summary"])
    rag_chain = prompt | llm_used
    raw = rag_chain.invoke({"context": context, "question": query})
    answer_text = raw.strip() if isinstance(raw, str) else str(raw)
    answer_text = clean_text(answer_text)

    # Post-processing and safety checks
    if "not in knowledge base" in answer_text.lower():
        answer_text = "Not in knowledge base."
    else:
        answer_text = enforce_min_len(answer_text, min_chars=10)
    
    # Recommendation: Use different character limits for QA and summary
    char_limit = SUMMARY_MAX_CHAR_LEN_RESP if query_meta["is_summary"] else MAX_CHAR_LEN_RESP
    answer_text = enforce_char_limit(answer_text, char_limit)

    result = {"answer": answer_text, "sources": list(sources)}
    if return_context:
        result["context"] = context
        result["doc_sources"] = [d.metadata.get("source", "Unknown") for d in docs]
    return result
