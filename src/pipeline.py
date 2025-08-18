"""
Pipeline nodes and orchestration for RAG pipeline.
"""
import re
from functools import lru_cache
from langchain_core.prompts import PromptTemplate

from src.llm import get_llm
from src.vectorstore import VectorStoreManager
from src.config import (
    TOP_K_RESULTS, TOXIC_KEYWORDS, PROFANITY_WORDS,
    MAX_CHAR_LEN_RESP, KNOWLEDGE_BASE_NOT_LOADED, NO_DOCS_FOUND,
    PROMPT_TEMPLATE, PER_DOC_CHARS_ALLOWED, OVERALL_CHARS_ALLOWED,
)

# Optional compression flags (safe defaults if not present in config)
try:
    from src.config import USE_CONTEXT_COMPRESSION, COMPRESSION_MAX_SENTENCE_CHARS, COMPRESSION_TOPK, RETRIEVAL_FETCH_K
except Exception:
    USE_CONTEXT_COMPRESSION = False
    COMPRESSION_MAX_SENTENCE_CHARS = 200
    COMPRESSION_TOPK = 8
    RETRIEVAL_FETCH_K = 50

# -------- helpers --------
"""
enforce_min_len(ans, min_chars=10)
If the model's answer is too short (likely junk), replace it with the canonical 
refusal “Not in knowledge base.”
"""
def enforce_min_len(ans: str, min_chars: int = 10) -> str:
    return ans if len((ans or "").strip()) >= min_chars else "Not in knowledge base."

"""
enforce_char_limit(text, limit=MAX_CHAR_LEN_RESP)
Hard-caps the answer length to your character budget; trims at a word boundary 
and appends an ellipsis if needed.
"""
def enforce_char_limit(text: str, limit: int = MAX_CHAR_LEN_RESP) -> str:
    s = " ".join((text or "").split()).strip()
    if len(s) <= limit: return s
    cut = s[: max(0, limit - 1)]
    if " " in cut: cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "…"

"""
classify_query(text: str)
If the query contains toxic or profanity keywords, classify it accordingly.
"""
def classify_query(text: str):
    lower = (text or "").lower()
    flags = set()
    if any(w in lower for w in TOXIC_KEYWORDS): flags.add("toxic")
    if any(w in lower for w in PROFANITY_WORDS): flags.add("profanity")
    return {"is_safe": not bool(flags), "flags": sorted(flags)}

"""
Clean the text by removing unnecessary whitespace and formatting.
"""
LIST_BULLET_RE = re.compile(r'^\s*(?:[-–—•*]\s+)+', re.M)
SEP_LINE_RE    = re.compile(r'^\s*---\s*$', re.M)
def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = SEP_LINE_RE.sub(" ", s)
    s = LIST_BULLET_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

"""
Build the context for the LLM by chunking the documents.
"""
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
def build_context(docs, per_doc_chars, overall_chars):
    parts, budget = [], overall_chars
    for d in docs:
        chunk = (d.page_content or "")[:per_doc_chars]
        sentences = [s.strip() for s in SENT_SPLIT.split(chunk) if s.strip()]
        safe, used = [], 0
        for s in sentences:
            if used + len(s) + 1 > per_doc_chars: break
            safe.append(s); used += len(s) + 1
        snippet = " ".join(safe).strip() or chunk.strip()
        snippet = clean_text(snippet)
        add = snippet[:budget].rstrip()
        if add:
            parts.append(add)
            budget -= len(add) + 2
            if budget <= 0: break
    return "\n\n".join(parts).strip()

# -------- query-focused compression using the SAME QA LLM --------
@lru_cache(maxsize=1)
def _qa_llm():
    return get_llm()

"""
Defining the extraction prompt for the QA LLM.
"""

EXTRACT_PROMPT = PromptTemplate.from_template(
    "You are an information extractor.\n"
    "Question: {question}\n"
    "Passage: {passage}\n\n"
    f"Return ONE sentence (<= {COMPRESSION_MAX_SENTENCE_CHARS} chars) that best answers the question.\n"
    "If the passage does not contain the answer, return exactly: NONE\n"
    "Answer:"
)

"""
Compress the snippets using the QA LLM.
"""
def compress_snippets(docs, question: str, per_doc_chars: int, overall_chars: int):
    llm = _qa_llm()
    chain = EXTRACT_PROMPT | llm

    evidences = []
    for d in docs:
        passage = (d.page_content or "")[:per_doc_chars]
        out = chain.invoke({"question": question, "passage": passage})
        line = clean_text(out if isinstance(out, str) else str(out))
        if not line or line.upper() == "NONE":
            continue
        if len(line) > COMPRESSION_MAX_SENTENCE_CHARS:
            line = line[:COMPRESSION_MAX_SENTENCE_CHARS-1].rsplit(" ", 1)[0].rstrip() + "…"
        evidences.append(line)

    # deduplicate (case-insensitive) and keep top-N
    uniq, seen = [], set()
    for s in evidences:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
        if len(uniq) >= COMPRESSION_TOPK:
            break

    # join within overall budget
    ctx_parts, used = [], 0
    for s in uniq:
        if used + len(s) + 1 > overall_chars: break
        ctx_parts.append(s); used += len(s) + 1
    return "\n".join(ctx_parts).strip()

# -------- main RAG function (no summarizer refs) --------
def retrieve_and_generate(manager, llm, query: str, return_context: bool = False) -> dict:
    if manager.vector_store is None:
        if not manager.load():
            base = {"answer": KNOWLEDGE_BASE_NOT_LOADED, "sources": []}
            return {**base, "context": ""} if return_context else base

    retriever = manager.vector_store.as_retriever(
        search_kwargs={"k": TOP_K_RESULTS, "fetch_k": RETRIEVAL_FETCH_K, "mmr": True, "lambda_mult": 0.4}
    )
    docs = retriever.invoke(query)
    if not docs:
        base = {"answer": NO_DOCS_FOUND, "sources": []}
        return {**base, "context": ""} if return_context else base

    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    PER_DOC_CHARS = PER_DOC_CHARS_ALLOWED
    OVERALL_CHARS = OVERALL_CHARS_ALLOWED

    if USE_CONTEXT_COMPRESSION:
        context = compress_snippets(docs, query, PER_DOC_CHARS, OVERALL_CHARS) or \
                  build_context(docs, PER_DOC_CHARS, OVERALL_CHARS)
    else:
        context = build_context(docs, PER_DOC_CHARS, OVERALL_CHARS)

    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    llm_used = llm or _qa_llm()
    rag_chain = prompt | llm_used
    raw = rag_chain.invoke({"context": context, "question": query})
    answer_text = raw.strip() if isinstance(raw, str) else str(raw)
    answer_text = clean_text(answer_text)

    # Canonicalize abstention + length controls
    if "not in knowledge base" in answer_text.lower():
        answer_text = "Not in knowledge base."
    else:
        answer_text = enforce_min_len(answer_text, min_chars=10)
    answer_text = enforce_char_limit(answer_text, MAX_CHAR_LEN_RESP)

    result = {"answer": answer_text, "sources": list(sources)}
    if return_context:
        result["context"] = context
        result["doc_sources"] = [d.metadata.get("source", "Unknown") for d in docs]
    return result
