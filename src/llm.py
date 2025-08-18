"""
LLM loading and answer generation for RAG pipeline.
Uncomment def get_summarizer(): if want to use summarizer also from config
"""
import logging
from src.config import (
    LLM_REPO_ID, LLM_TASK, HUGGINGFACEHUB_API_TOKEN, LLM_TEMPERATURE, LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_ID, LOCAL_LLM_TASK, LOCAL_LLM_MAX_NEW_TOKENS,
    # SUMMARIZER_MODEL_ID, SUMMARIZER_TASK, SUMMARIZER_MAX_LEN, SUMMARIZER_MIN_LEN
)

def get_llm(force_local: bool = True):
    if not force_local:
        api_token = HUGGINGFACEHUB_API_TOKEN
        if api_token:
            try:
                from langchain_huggingface import HuggingFaceEndpoint
                return HuggingFaceEndpoint(
                    repo_id=LLM_REPO_ID,
                    task=LLM_TASK,                        # e.g., "text-generation"
                    temperature=LLM_TEMPERATURE,
                    max_new_tokens=LLM_MAX_NEW_TOKENS,
                    huggingfacehub_api_token=api_token,
                    provider="hf-inference",
                    return_full_text=False               # Don't return full text
                )
            except Exception as e:
                logging.warning(f"HF endpoint failed, falling back to local model: {e}")
    """
    Robust get_llm:
    - Try Hugging Face Inference endpoint (if HUGGINGFACEHUB_API_TOKEN is set).
    - If that fails or token is missing, fall back to a local model pipeline.
    """
    # LOCAL FALLBACK (Local model)
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    tok = AutoTokenizer.from_pretrained(LOCAL_LLM_ID)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_LLM_ID)

    tok.model_max_length = 512 # Optional safety:
    gen_pipe = pipeline(
        LOCAL_LLM_TASK,
        model=mdl,
        tokenizer=tok,
        max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        truncation=True,
        min_new_tokens=8,
        no_repeat_ngram_size=3,   # optional but helpful
    )

    logging.info(f"Using local model {LOCAL_LLM_ID} for LLM (fallback mode)")
    return HuggingFacePipeline(pipeline=gen_pipe)

# OPTIONAL_SUMMARIZER = False
# def get_summarizer():
#     """
#     Local summarizer pipeline (Pegasus/BART). Returns a LangChain HuggingFacePipeline.
#     """
#     from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
#     from langchain_huggingface import HuggingFacePipeline
#     tok = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_ID)
#     mdl = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_ID)
#     gen_pipe = pipeline(
#         SUMMARIZER_TASK,               # <-- 'summarization'
#         model=mdl,
#         tokenizer=tok,
#         truncation=True,
#         min_length=SUMMARIZER_MIN_LEN,
#         max_length=SUMMARIZER_MAX_LEN,
#         do_sample=False,
#         num_beams=4,                   # a bit more polish while staying deterministic
#         no_repeat_ngram_size=3
#     )
#     logging.info(f"Using local summarizer {SUMMARIZER_MODEL_ID}")
#     return HuggingFacePipeline(pipeline=gen_pipe)
