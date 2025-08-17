"""
LLM loading and answer generation for RAG pipeline.
"""
import os
import logging
from src.config import LLM_REPO_ID, LLM_TASK, HUGGINGFACEHUB_API_TOKEN

def get_llm(force_local: bool = True):
    if not force_local:
        api_token = HUGGINGFACEHUB_API_TOKEN
        if api_token:
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                return HuggingFaceEndpoint(
                    repo_id=LLM_REPO_ID,
                    task=LLM_TASK,
                    temperature=0.1,
                    max_new_tokens=512,
                    huggingfacehub_api_token=api_token,
                    provider="hf-inference"
                )
            except Exception as e:
                logging.warning(f"HF endpoint failed, falling back to local model: {e}")
    """
    Robust get_llm:
    - Try Hugging Face Inference endpoint (if HUGGINGFACE_API_TOKEN is set).
    - If that fails or token is missing, fall back to a local FLAN‑T5 pipeline.
    """
    # LOCAL FALLBACK (FLAN-T5)
    # Local fallback
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    local_model = os.getenv("LOCAL_LLM_ID", "google/flan-t5-base")
    tok = AutoTokenizer.from_pretrained(local_model)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(local_model)
    gen_pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, max_new_tokens=256)
    logging.info("Using local FLAN-T5 model for LLM (fallback mode)")
    return HuggingFacePipeline(pipeline=gen_pipe)
