"""
LLM loading and answer generation for RAG pipeline.
"""
import os
import logging
from src.config import LLM_REPO_ID, LLM_TASK, HUGGINGFACEHUB_API_TOKEN, LOCAL_LLM_ID, LOCAL_LLM_TASK, LOCAL_LLM_MAX_NEW_TOKENS

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
    - If that fails or token is missing, fall back to a local model pipeline.
    """
    # LOCAL FALLBACK (FLAN-T5)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    tok = AutoTokenizer.from_pretrained(LOCAL_LLM_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_ID)
    gen_pipe = pipeline(
        LOCAL_LLM_TASK,
        model=mdl,
        tokenizer=tok,
        max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS
    )
    logging.info(f"Using local model {LOCAL_LLM_ID} for LLM (fallback mode)")
    return HuggingFacePipeline(pipeline=gen_pipe)
