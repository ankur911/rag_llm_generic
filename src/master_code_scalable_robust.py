"""
Unified and Scalable RAG Pipeline with a Persistent Vector Store.
Robust version: FAISS→Chroma fallback, HF Inference→local FLAN-T5 fallback, trimmed context, FastAPI endpoints, LangGraph workflow.
"""
import os
import logging
from typing import List, TypedDict, Dict, Any, Optional
from collections.abc import Iterable
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================
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

# ==============================================================================
# SECTION 2: UTILITIES & LAZY LOADING
# ==============================================================================
_models: Dict[str, Any] = {}

def _load_model(key: str, loader):
    if key not in _models:
        _models[key] = loader()
    return _models[key]

def get_embedding_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return _load_model("embedding", lambda: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

def get_llm():
    """
    Robust get_llm:
    - Try Hugging Face Inference endpoint (if HUGGINGFACEHUB_API_TOKEN is set).
    - If that fails or token is missing, fall back to a local FLAN‑T5 pipeline.
    """
    load_dotenv()
    # Set this flag to True to always use local FLAN-T5 model (for testing fallback)
    FORCE_LOCAL_LLM = True  # <-- Set to False to enable HuggingFace remote model

    if not FORCE_LOCAL_LLM:
        api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if api_token:
            try:
                # REMOTE HUGGINGFACE ENDPOINT (commented out for fallback testing)
                # from langchain_huggingface import HuggingFaceEndpoint
                # return _load_model(
                #     "llm",
                #     lambda: HuggingFaceEndpoint(
                #         repo_id=LLM_REPO_ID,
                #         task=LLM_TASK,
                #         temperature=0.1,
                #         max_new_tokens=512,
                #         huggingfacehub_api_token=api_token,
                #         provider="hf-inference"
                #     ),
                # )
                pass
            except Exception as e:
                logging.warning(f"HF endpoint failed, falling back to local model: {e}")

    # LOCAL FALLBACK (FLAN-T5)
    # This branch will always run if FORCE_LOCAL_LLM is True
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    local_model = os.getenv("LOCAL_LLM_ID", "google/flan-t5-base")
    tok = AutoTokenizer.from_pretrained(local_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(local_model)
    gen_pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)
    logging.info("Using local FLAN-T5 model for LLM (fallback mode)")
    return _load_model("llm", lambda: HuggingFacePipeline(pipeline=gen_pipe))

# ==============================================================================
# SECTION 3: VECTOR STORE MANAGEMENT (FAISS -> Chroma fallback)
# ==============================================================================
_USE_FAISS = True
try:
    import faiss
except Exception:
    _USE_FAISS = False
    logging.warning("FAISS not available; falling back to Chroma vector store.")

if _USE_FAISS:
    from langchain_community.vectorstores import FAISS as VS
else:
    from langchain_community.vectorstores import Chroma as VS

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def normalize_sources(sources: Any) -> List[str]:
    if sources is None:
        return []
    if isinstance(sources, str):
        return [sources]
    if isinstance(sources, Iterable):
        return [str(s).strip() for s in sources if s]
    raise TypeError("sources must be a str or an iterable of str")

class VectorStoreManager:
    def __init__(self, path: str = VECTOR_STORE_PATH):
        self.path = path
        self.embedding_model = get_embedding_model()
        self.vector_store: Optional[Any] = None

    def _get_loader(self, source: str):
        if source.lower().endswith(".pdf") and (source.startswith("/") or "://" not in source):
            return PyPDFLoader(source)
        if source.startswith("http://") or source.startswith("https://"):
            return WebBaseLoader([source])
        logging.warning(f"No loader available for source: {source}")
        return None

    def build_and_save(self, sources: Any) -> bool:
        logging.info("Building vector store...")
        sources = normalize_sources(sources)
        if not sources:
            logging.error("No sources provided. Vector store not built.")
            return False
        docs = []
        for src in sources:
            loader = self._get_loader(src)
            if not loader:
                continue
            try:
                loaded = loader.load()
                docs.extend(loaded)
                logging.info(f"Loaded {len(loaded)} docs from {src}")
            except Exception as e:
                logging.error(f"Failed to load source {src}: {e}", exc_info=True)
        if not docs:
            logging.error("No documents were loaded. Vector store not built.")
            return False
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        logging.info(f"Split {len(docs)} raw docs into {len(chunks)} chunks.")
        if _USE_FAISS:
            self.vector_store = VS.from_documents(chunks, self.embedding_model)
            self.vector_store.save_local(self.path)
        else:
            self.vector_store = VS.from_documents(chunks, self.embedding_model, persist_directory=self.path)
            self.vector_store.persist()
        logging.info(f"Vector store built and saved to {self.path}")
        return True

    def load(self) -> bool:
        try:
            if self.vector_store is not None:
                return True
            if not os.path.exists(self.path):
                logging.warning(f"Vector store path '{self.path}' does not exist.")
                return False
            if _USE_FAISS:
                from langchain_community.vectorstores import FAISS
                self.vector_store = FAISS.load_local(
                    self.path, self.embedding_model, allow_dangerous_deserialization=True
                )
            else:
                from langchain_community.vectorstores import Chroma
                self.vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=self.path
                )
            logging.info("Vector store loaded from disk.")
            return True
        except Exception as e:
            logging.error(f"Error loading vector store: {e}", exc_info=True)
            self.vector_store = None
            return False

# ==============================================================================
# SECTION 4: PIPELINE NODES & UTILS
# ==============================================================================
from langchain_core.prompts import PromptTemplate

def enforce_char_limit(text: str, limit: int = 140) -> str:
    s = " ".join((text or "").split()).strip()
    if len(s) <= limit:
        return s
    cut = s[: max(0, limit - 1)]
    if " " in cut:
        cut = cut[: cut.rfind(" ")].rstrip()
    return cut + "…"

def classify_query(text: str) -> Dict[str, Any]:
    lower = text.lower()
    flags = set()
    if any(w in lower for w in TOXIC_KEYWORDS):
        flags.add("toxic")
    if any(w in lower for w in PROFANITY_WORDS):
        flags.add("profanity")
    return {"is_safe": not bool(flags), "flags": sorted(flags)}

def retrieve_and_generate(query: str) -> Dict[str, Any]:
    if vector_store_manager.vector_store is None:
        if not vector_store_manager.load():
            return {"answer": "Knowledge base not loaded.", "sources": []}
    try:
        retriever = vector_store_manager.vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
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
        llm = get_llm()
        rag_chain = prompt | llm
        raw = rag_chain.invoke({"context": context, "question": query})
        answer_text = raw.strip() if isinstance(raw, str) else str(raw)
        answer_text = enforce_char_limit(answer_text, 140)
        return {"answer": answer_text, "sources": list(sources)}
    except Exception as e:
        logging.error(f"Error during RAG generation: {e}", exc_info=True)
        raise

vector_store_manager = VectorStoreManager()

# ==============================================================================
# SECTION 5: LANGGRAPH WORKFLOW
# ==============================================================================
from langgraph.graph import StateGraph, END
class RAGState(TypedDict):
    query: str
    is_safe: bool
    final_answer: Dict[str, Any]

def classify_node(state: RAGState) -> Dict[str, Any]:
    out = classify_query(state["query"])
    return {"is_safe": out["is_safe"]}

def generate_node(state: RAGState) -> Dict[str, Any]:
    if not state["is_safe"]:
        return {"final_answer": {"answer": "Query flagged as unsafe.", "sources": []}}
    return {"final_answer": retrieve_and_generate(state["query"])}

def make_rag_app():
    builder = StateGraph(RAGState)
    builder.add_node("classify", classify_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("classify")
    builder.add_edge("classify", "generate")
    return builder.compile()

rag_app = make_rag_app()

# ==============================================================================
# SECTION 6: FASTAPI APPLICATION
# ==============================================================================
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("--- Application Startup ---")
    vector_store_manager.load()
    yield
    logging.info("--- Application Shutdown ---")

app = FastAPI(title="Scalable RAG API (Robust)", version="3.3.0", lifespan=lifespan)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@app.get("/health")
async def health():
    status = {
        "vector_store_loaded": vector_store_manager.vector_store is not None,
        "vector_store_path_exists": os.path.exists(VECTOR_STORE_PATH),
        "embedding_model": EMBEDDING_MODEL,
        "llm_repo": LLM_REPO_ID,
        "backend": "FAISS" if _USE_FAISS else "Chroma",
    }
    return status

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        result = await rag_app.ainvoke({"query": request.query})
        final = result.get("final_answer", {})
        if not final:
            final = {"answer": "No answer generated.", "sources": []}
        return QueryResponse(**final)
    except Exception as e:
        logging.error(f"Error processing /query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ==============================================================================
# SECTION 7: ONE-TIME VECTOR STORE BUILDER
# ==============================================================================
if __name__ == "__main__":
    import sys
    load_dotenv()
    print("--- Running Vector Store Builder (Robust) ---")
    ok = VectorStoreManager().build_and_save(DOCUMENT_SOURCES)
    if ok:
        print("✅ Vector store built successfully.")
    else:
        print("❌ Failed to build vector store. Check logs above.")
    print("\nTo run the API server, use:")
    print("uvicorn src.master_code_scalable_robust:app --reload")

    # PLAN-A: Direct test (no server)
    if len(sys.argv) > 1 and sys.argv[1] == "plan-a":
        print("\n--- PLAN-A: Direct RAG test ---")
        query = "List key WHO recommendations for infant and young child feeding."
        vector_store_manager.load()
        result = retrieve_and_generate(query)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")

    # PLAN-B: Server/API test
    elif len(sys.argv) > 1 and sys.argv[1] == "plan-b":
        print("\n--- PLAN-B: API test via FastAPI server ---")
        import subprocess, time, requests, json
        # Start server
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", "uvicorn", "src.master_code_scalable_robust:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"]
        server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        print("🚀 Starting Uvicorn server…")
        # Wait for server startup
        started = False
        t0 = time.time()
        while time.time() - t0 < 30:
            line = server.stdout.readline()
            if line:
                print(line, end="")
                if "Application startup complete" in line:
                    started = True
                    break
            elif server.poll() is not None:
                break
        if started:
            print("\nServer started. Running health and query tests...")
            try:
                r = requests.get("http://127.0.0.1:8000/health", timeout=10)
                print("/health:", r.status_code, json.dumps(r.json(), indent=2))
                payload = {"query": "List key WHO recommendations for infant and young child feeding."}
                r = requests.post("http://127.0.0.1:8000/query", json=payload, timeout=120)
                print("/query:", r.status_code, json.dumps(r.json(), indent=2))
            except Exception as e:
                print("API test failed:", e)
            finally:
                server.terminate()
                server.wait(timeout=10)
                print("🛑 Server stopped.")
        else:
            print("❌ Server did not start.")
