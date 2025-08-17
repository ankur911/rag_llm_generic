"""
FastAPI app for RAG pipeline.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import EMBEDDING_MODEL, LLM_REPO_ID, VECTOR_STORE_PATH
from src.vectorstore import VectorStoreManager
from src.llm import get_llm
from src.pipeline import retrieve_and_generate

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: list

manager = VectorStoreManager()
llm = get_llm()
app = FastAPI(title="Scalable RAG API (Robust)", version="3.3.0")

@app.get("/health")
async def health():
    status = {
        "vector_store_loaded": manager.vector_store is not None,
        "vector_store_path_exists": VECTOR_STORE_PATH,
        "embedding_model": EMBEDDING_MODEL,
        "llm_repo": LLM_REPO_ID,
        "backend": "FAISS",
    }
    return status

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = retrieve_and_generate(manager, llm, request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
