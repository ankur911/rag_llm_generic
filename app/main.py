import sys
from src.config import DOCUMENT_SOURCES
from src.vectorstore import VectorStoreManager
from src.llm import get_llm
from src.pipeline import retrieve_and_generate

def build_or_load_store() -> VectorStoreManager:
    manager = VectorStoreManager()
    print("Embedding model in use:", getattr(manager.embedding_model, "model_name", "unknown"))
    if not manager.load():
        print("--- Running Vector Store Builder (Robust) ---")
        ok = manager.build_and_save(DOCUMENT_SOURCES)
        if ok:
            print("✅ Vector store built successfully.")
        else:
            print("❌ Failed to build vector store. Check logs above.")
    else:
        print("✅ Vector store loaded.")
    return manager

if __name__ == "__main__":
    manager = build_or_load_store()
    llm = get_llm()

    if len(sys.argv) > 1 and sys.argv[1] == "plan-a":
        print("\n--- PLAN-A: Direct RAG test ---")
        query = "List key WHO recommendations for infant and young child feeding."
        result = retrieve_and_generate(manager, llm, query, return_context=True)
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("\n--- Context used ---")
        print(result.get("context", ""))

    elif len(sys.argv) > 1 and sys.argv[1] == "plan-b":
        print("\n--- PLAN-B: API test via FastAPI server ---")
        import subprocess, time, requests, json, os
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"]
        server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        print("🚀 Starting Uvicorn server…")
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
