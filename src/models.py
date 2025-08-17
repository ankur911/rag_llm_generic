"""
Data models and types for RAG pipeline.
"""
from typing import TypedDict, Dict, Any, List

class RAGState(TypedDict):
    query: str
    is_safe: bool
    final_answer: Dict[str, Any]
