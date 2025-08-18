"""
Pipeline nodes and orchestration for RAG pipeline.
"""
from src.config import TOP_K_RESULTS, TOXIC_KEYWORDS, PROFANITY_WORDS, MAX_CHAR_LEN_RESP, KNOWLEDGE_BASE_NOT_LOADED, NO_DOCS_FOUND, PROMPT_TEMPLATE, PER_DOC_CHARS_ALLOWED, OVERALL_CHARS_ALLOWED
from langchain_core.prompts import PromptTemplate
from src.vectorstore import VectorStoreManager
from src.llm import get_llm

def enforce_char_limit(text: str, limit: int = MAX_CHAR_LEN_RESP) -> str:
    s = " ".join((text or "").split()).strip()
    if len(s) <= limit:
        return s
    cut = s[: max(0, limit - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "…"

def classify_query(text: str):
    lower = text.lower()
    flags = set()
    if any(w in lower for w in TOXIC_KEYWORDS):
        flags.add("toxic")
    if any(w in lower for w in PROFANITY_WORDS):
        flags.add("profanity")
    return {"is_safe": not bool(flags), "flags": sorted(flags)}

def retrieve_and_generate(manager, llm, query: str) -> dict:
    if manager.vector_store is None:
        if not manager.load():
            return {"answer": KNOWLEDGE_BASE_NOT_LOADED, "sources": []}
    retriever = manager.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    docs = retriever.invoke(query)
    if not docs:
        return {"answer": NO_DOCS_FOUND, "sources": []}
    PER_DOC_CHARS = PER_DOC_CHARS_ALLOWED
    OVERALL_CHARS = OVERALL_CHARS_ALLOWED
    context = "\n---\n".join([d.page_content[:PER_DOC_CHARS] for d in docs])[:OVERALL_CHARS]
    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    rag_chain = prompt | llm
    raw = rag_chain.invoke({"context": context, "question": query})
    answer_text = raw.strip() if isinstance(raw, str) else str(raw)
    answer_text = enforce_char_limit(answer_text, MAX_CHAR_LEN_RESP)
    return {"answer": answer_text, "sources": list(sources)}

# --- add a flag + return context/doc metadata ---
# def retrieve_and_generate(manager, llm, query: str, return_docs: bool = False) -> dict:
#     if manager.vector_store is None:
#         if not manager.load():
#             return {"answer": KNOWLEDGE_BASE_NOT_LOADED, "sources": []}

#     retriever = manager.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
#     docs = retriever.invoke(query)
#     if not docs:
#         base = {"answer": NO_DOCS_FOUND, "sources": []}
#         return {**base, "context": "", "doc_sources": []} if return_docs else base

#     PER_DOC_CHARS = PER_DOC_CHARS_ALLOWED
#     OVERALL_CHARS = OVERALL_CHARS_ALLOWED
#     context_chunks = [d.page_content[:PER_DOC_CHARS] for d in docs]
#     context = "\n---\n".join(context_chunks)[:OVERALL_CHARS]
#     sources = sorted({d.metadata.get("source", "Unknown") for d in docs})

#     prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
#     rag_chain = prompt | llm
#     raw = rag_chain.invoke({"context": context, "question": query})
#     answer_text = raw.strip() if isinstance(raw, str) else str(raw)
#     answer_text = enforce_char_limit(answer_text, MAX_CHAR_LEN_RESP)

#     result = {"answer": answer_text, "sources": list(sources)}
#     if return_docs:
#         result["context"] = context
#         result["doc_sources"] = [d.metadata.get("source", "Unknown") for d in docs]
#     return result

