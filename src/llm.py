"""
LLM loading and answer generation for the RAG pipeline.

Behavior:
- If force_local is True  -> always local
- If force_local is False -> always remote (no fallback)
- Else uses USE_REMOTE_LLM; if True, tries remote (endpoint_url if set, else serverless).
  On remote failure, falls back to local.
"""

import logging
from typing import Optional

# Import the config module robustly, then read attributes
from src import config as _cfg

# --- Config values with safe defaults ---
USE_REMOTE_LLM = getattr(_cfg, "USE_REMOTE_LLM", False)
LLM_REPO_ID = getattr(_cfg, "LLM_REPO_ID", "google/flan-t5-base")
LLM_TASK = getattr(_cfg, "LLM_TASK", "text2text-generation")
HUGGINGFACEHUB_API_TOKEN = getattr(_cfg, "HUGGINGFACEHUB_API_TOKEN", "")

# Optional: dedicated Inference Endpoint URL (if you create one)
HF_ENDPOINT_URL = getattr(_cfg, "HF_ENDPOINT_URL", None)

LLM_TEMPERATURE = float(getattr(_cfg, "LLM_TEMPERATURE", 0.0))
LLM_MAX_NEW_TOKENS = int(getattr(_cfg, "LLM_MAX_NEW_TOKENS", 256))

LOCAL_LLM_ID = getattr(_cfg, "LOCAL_LLM_ID", "bigscience/mt0-base")
LOCAL_LLM_TASK = getattr(_cfg, "LOCAL_LLM_TASK", "text2text-generation")
LOCAL_LLM_MAX_NEW_TOKENS = int(getattr(_cfg, "LOCAL_LLM_MAX_NEW_TOKENS", 128))


# --------------------- builders ---------------------

def _build_local_llm():
    """Create a deterministic local generation pipeline (seq2seq or decoder-only)."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
        pipeline,
    )
    from langchain_huggingface import HuggingFacePipeline

    mdl_cls = AutoModelForSeq2SeqLM if LOCAL_LLM_TASK == "text2text-generation" else AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(LOCAL_LLM_ID)
    try:
        tok.model_max_length = min(getattr(tok, "model_max_length", 512), 512)
    except Exception:
        pass

    mdl = mdl_cls.from_pretrained(LOCAL_LLM_ID)
    gen_pipe = pipeline(
        LOCAL_LLM_TASK,
        model=mdl,
        tokenizer=tok,
        max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
        do_sample=False,        # deterministic for evals
        num_beams=1,
        truncation=True,
        min_new_tokens=8,
        no_repeat_ngram_size=3,
    )
    logging.info(f"Using LOCAL model: {LOCAL_LLM_ID} ({LOCAL_LLM_TASK})")
    return HuggingFacePipeline(pipeline=gen_pipe)


def _build_remote_serverless():
    """Create a remote HF Serverless client (router)."""
    from langchain_huggingface import HuggingFaceEndpoint
    return HuggingFaceEndpoint(
        repo_id=LLM_REPO_ID,
        task=LLM_TASK,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        provider="hf-inference",          # serverless router
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        return_full_text=False,           # avoid prompt echo (decoder-only)
    )


def _build_remote_endpoint():
    """Create a remote HF *dedicated* Inference Endpoint client (if URL is provided)."""
    from langchain_huggingface import HuggingFaceEndpoint
    if not HF_ENDPOINT_URL:
        raise RuntimeError("HF_ENDPOINT_URL not set; cannot build remote endpoint client.")
    return HuggingFaceEndpoint(
        endpoint_url=HF_ENDPOINT_URL,
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        task=LLM_TASK,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        return_full_text=False,
    )


# --------------------- public API ---------------------

def get_llm(force_local: Optional[bool] = None):
    """
    Returns a LangChain-compatible LLM with the routing rules described above.
    """
    # 1) Explicitly forced LOCAL
    if force_local is True:
        return _build_local_llm()

    # 2) Explicitly forced REMOTE (no fallback)
    if force_local is False:
        if not HUGGINGFACEHUB_API_TOKEN:
            raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is not set for remote LLM.")
        llm = _build_remote_endpoint() if HF_ENDPOINT_URL else _build_remote_serverless()
        logging.info(f"Using REMOTE model (forced): {HF_ENDPOINT_URL or LLM_REPO_ID} ({LLM_TASK})")
        return llm

    # 3) Config-driven path
    if USE_REMOTE_LLM:
        if not HUGGINGFACEHUB_API_TOKEN:
            logging.warning("USE_REMOTE_LLM=True but no HF token found; using LOCAL model.")
            return _build_local_llm()
        try:
            llm = _build_remote_endpoint() if HF_ENDPOINT_URL else _build_remote_serverless()
            # Preflight probe: if router/endpoint 404/403/etc., fall back to local
            _ = llm.invoke("ping")
            logging.info(f"Using REMOTE model: {HF_ENDPOINT_URL or LLM_REPO_ID} ({LLM_TASK})")
            return llm
        except Exception as e:
            logging.warning(f"Remote LLM unavailable ({HF_ENDPOINT_URL or LLM_REPO_ID}): {e}. Falling back to LOCAL.")
            return _build_local_llm()

    # Default: LOCAL
    return _build_local_llm()
