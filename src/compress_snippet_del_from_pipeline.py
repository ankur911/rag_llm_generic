"""
This file contains the LLM-based context compression logic that was
previously part of the main pipeline. It uses the main QA LLM to extract
relevant sentences from retrieved documents, providing an alternative
to the embed-ranking compression method.
"""
import re
from functools import lru_cache
from langchain_core.prompts import PromptTemplate

from src.llm import get_llm
from src.config import (
    COMPRESSION_MAX_SENTENCE_CHARS,
    COMPRESSION_TOPK,
)

# --- Text processing helpers ---

LIST_BULLET_RE = re.compile(r'^\s*(?:[-–—•*]\s+)+', re.M)
SEP_LINE_RE = re.compile(r'^\s*---\s*$', re.M)

def clean_text(s: str) -> str:
    """Remove redundant whitespace, list bullets, and separator lines."""
    s = (s or "").strip()
    s = SEP_LINE_RE.sub(" ", s)
    s = LIST_BULLET_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# --- LLM-based context compression ---

@lru_cache(maxsize=1)
def _qa_llm():
    """Cached LLM instance for reuse in compression."""
    return get_llm()

EXTRACT_PROMPT = PromptTemplate.from_template(
    "You are an information extractor.\n"
    "Question: {question}\n"
    "Passage: {passage}\n\n"
    f"Return ONE sentence (<= {COMPRESSION_MAX_SENTENCE_CHARS} chars) that best answers the question.\n"
    "If the passage does not contain the answer, return exactly: NONE\n"
    "Answer:"
)

def compress_snippets_with_llm(docs, question: str, per_doc_chars: int, overall_chars: int):
    """
    Compresses document snippets by using the main QA LLM to extract the single
    most relevant sentence from each document chunk.
    """
    llm = _qa_llm()
    chain = EXTRACT_PROMPT | llm

    evidences = []
    for d in docs:
        passage = (d.page_content or "")[:per_doc_chars]
        out = chain.invoke({"question": question, "passage": passage})
        line = clean_text(out if isinstance(out, str) else str(out))
        if not line or line.upper() == "NONE":
            continue
        # Enforce character limit on the extracted sentence
        if len(line) > COMPRESSION_MAX_SENTENCE_CHARS:
            line = line[:COMPRESSION_MAX_SENTENCE_CHARS-1].rsplit(" ", 1)[0].rstrip() + "…"
        evidences.append(line)

    # Deduplicate (case-insensitive) and keep top-N
    uniq, seen = [], set()
    for s in evidences:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
        if len(uniq) >= COMPRESSION_TOPK:
            break

    # Join the unique, extracted sentences within the overall character budget
    ctx_parts, used = [], 0
    for s in uniq:
        if used + len(s) + 1 > overall_chars:
            break
        ctx_parts.append(s)
        used += len(s) + 1
    return "\n".join(ctx_parts).strip()
