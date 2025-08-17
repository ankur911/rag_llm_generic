"""
Embedding logic for RAG pipeline.
"""
from src.config import EMBEDDING_MODEL
def get_embedding_model(model_name: str):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)