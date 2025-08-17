"""
Pipeline nodes and orchestration for RAG pipeline.
"""
from src.config import TOP_K_RESULTS, TOXIC_KEYWORDS, PROFANITY_WORDS
from langchain_core.prompts import PromptTemplate
from src.vectorstore import VectorStoreManager
from src.llm import get_llm

def enforce_char_limit(text: str, limit: int = 140) -> str:
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
            return {"answer": "Knowledge base not loaded.", "sources": []}
    retriever = manager.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    docs = retriever.invoke(query)
    if not docs:
        return {"answer": "I couldn't retrieve relevant context from the knowledge base.", "sources": []}
    PER_DOC_CHARS = 900
    OVERALL_CHARS = 2200
    context = "\n---\n".join([d.page_content[:PER_DOC_CHARS] for d in docs])[:OVERALL_CHARS]
    sources = sorted({d.metadata.get("source", "Unknown") for d in docs})
    prompt = PromptTemplate.from_template(
        "You are a concise, factual assistant. Answer ONLY using the context.\n"
        "Your ENTIRE answer must be <= 140 characters.\n"
        "If the answer is not in the context, say so briefly.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (<=140 chars):"
    )
    rag_chain = prompt | llm
    raw = rag_chain.invoke({"context": context, "question": query})
    answer_text = raw.strip() if isinstance(raw, str) else str(raw)
    answer_text = enforce_char_limit(answer_text, 140)
    return {"answer": answer_text, "sources": list(sources)}
