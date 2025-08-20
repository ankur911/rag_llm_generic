"""
Test script to resolve and print LLM and embedding model parameters during pipeline orchestration (plan-a).
Allows flexible testing by overriding config parameters for different modes via command-line arguments.
"""
import os
import argparse
from src import config as _cfg
from src.vectorstore import VectorStoreManager
from src.llm import get_llm

def main(USE_REMOTE_LLM=False, USE_INSTRUCT_E5=False):
    # Override config parameters for testing
    _cfg.USE_REMOTE_LLM = USE_REMOTE_LLM
    # _cfg.HF_ENDPOINT_URL = HF_ENDPOINT_URL
    _cfg.USE_INSTRUCT_E5 = USE_INSTRUCT_E5

    print(f"[Test] USE_REMOTE_LLM: {USE_REMOTE_LLM}")
    # print(f"[Test] HF_ENDPOINT_URL: {HF_ENDPOINT_URL}")
    print(f"[Test] USE_INSTRUCT_E5: {USE_INSTRUCT_E5}")
    print("[Test] Embedding model from config:", _cfg.EMBEDDING_MODEL)
    print("[Test] LLM repo ID from config:", _cfg.LLM_REPO_ID)
    print("[Test] Local LLM ID from config:", _cfg.LOCAL_LLM_ID)

    manager = VectorStoreManager()
    print("[Test] Embedding model used in VectorStoreManager:", getattr(manager.embedding_model, "model_name", str(manager.embedding_model)))

    llm = get_llm()
    print("[Test] LLM object type:", type(llm))
    print("[Test] LLM details:", getattr(llm, "model_id", getattr(llm, "repo_id", str(llm))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-remote-llm", action="store_true", help="Use remote LLM")
    # parser.add_argument("--hf-endpoint-url", type=str, default=None, help="HuggingFace endpoint URL")
    parser.add_argument("--use-instruct-e5", action="store_true", help="Use instruct E5 embedding")
    args = parser.parse_args()

    main(
        USE_REMOTE_LLM=args.use_remote_llm,
        # HF_ENDPOINT_URL=args.hf_endpoint_url,
        USE_INSTRUCT_E5=args.use_instruct_e5
    )
