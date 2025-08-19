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

from src.config import (
    TOP_K_RESULTS,
    USE_CONTEXT_COMPRESSION,
    COMPRESSION_MAX_SENTENCE_CHARS,
    COMPRESSION_TOPK,
    RETRIEVAL_FETCH_K,
    RETRIEVAL_LAMBDA_MULT,
)

import math

# Optional compression flags (safe defaults if not present in config)
try:
    from src.config import USE_CONTEXT_COMPRESSION, COMPRESSION_MAX_SENTENCE_CHARS, COMPRESSION_TOPK, RETRIEVAL_FETCH_K
except Exception:
    USE_CONTEXT_COMPRESSION = False
    COMPRESSION_MAX_SENTENCE_CHARS = 200
    COMPRESSION_TOPK = 8
    RETRIEVAL_FETCH_K = 50

# -------- helpers --------

# Compute the cosine similarity between two vectors.
def _cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

# Remove duplicate lines while preserving order.
def _dedup_keep_order(lines):
    kept, seen = [], set()
    for s in lines:
        k = s.strip().lower()
        if k in seen: 
            continue
        seen.add(k); kept.append(s.strip())
    return kept

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
f"Build the context for the LLM by chunking the documents. This involves\n"
f"splitting the documents into smaller parts that fit within the model's\n"
f"context window of {TOKEN_LIMIT} for the used \"{get_llm.name()}\" from the config file"
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

"""
Build context using embedding rank - this is a more advanced method
that ranks sentences based on their relevance to the query by leveraging
sentence embeddings and cosine similarity.
"""

def build_context_embed_rank(
    manager,
    docs,
    question: str,
    per_doc_chars: int,
    overall_chars: int,
    top_sentences: int | None = None,
) -> str:
    """
    Build query-focused context by:
    1) collecting sentence candidates from doc chunks,
    2) filtering obvious meta/program lines,
    3) ranking by cosine to the query (same E5 embedder),
    4) taking top distinct sentences within budget.
    """
    # ---- config-derived caps ----
    TOPN = top_sentences or COMPRESSION_TOPK
    MAX_SENT_CHARS = COMPRESSION_MAX_SENTENCE_CHARS

    # 1) collect candidate sentences
    candidates = []
    for d in docs:
        chunk = (d.page_content or "")[:per_doc_chars]
        sentences = [s.strip() for s in SENT_SPLIT.split(chunk) if s.strip()]
        # basic length filter (avoid tiny fragments)
        sentences = [clean_text(s) for s in sentences if len(s) >= 30]
        # 2) filter meta/program lines
        ban = (
            "strategy", "programme", "program", "training", "course",
            "who provides", "who and unicef", "monitoring", "report", "launch"
        )
        sentences = [s for s in sentences if not any(b in s.lower() for b in ban)]
        # cap each sentence length for readability and to reduce prompt bloat
        sentences = [s[:MAX_SENT_CHARS].rstrip(" ,;:") for s in sentences]
        candidates.extend(sentences)

    if not candidates:
        return ""

    # 3) rank by cosine to query using SAME embedder (E5 adds prefixes internally)
    qv = manager.embedding_model.embed_query(question)
    sv = manager.embedding_model.embed_documents(candidates)
    scored = sorted(
        zip(candidates, sv),
        key=lambda p: _cosine(p[1], qv),  # cosine similarity
        reverse=True
    )

    # 4) keep top distinct lines and pack within overall budget
    top = _dedup_keep_order([c for c, _ in scored][:max(3, TOPN)])
    ctx, used = [], 0
    for s in top:
        # +1 accounts for the newline join below
        if used + len(s) + 1 > overall_chars:
            break
        ctx.append(s)
        used += len(s) + 1

    return "\n".join(ctx).strip()


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
    PER_DOC_CHARS = PER_DOC_CHARS_ALLOWED
    OVERALL_CHARS = OVERALL_CHARS_ALLOWED

    if USE_CONTEXT_COMPRESSION:
        context = build_context_embed_rank(
            manager, docs, query, PER_DOC_CHARS, OVERALL_CHARS, top_sentences=COMPRESSION_TOPK
        )
    if not context:
        context = build_context(docs, PER_DOC_CHARS, OVERALL_CHARS)
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
