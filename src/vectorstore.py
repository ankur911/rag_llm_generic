"""
Vector store management for RAG pipeline.
"""
import os
import logging
from src.config import VECTOR_STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.embedding import get_embedding_model
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    def __init__(self, path: str = VECTOR_STORE_PATH, embedding_model=None, model_name=None):
        self.path = path
        default_model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = embedding_model or get_embedding_model(default_model_name)
        self.vector_store = None

    def _get_loader(self, source: str):
        if source.lower().endswith(".pdf") and (source.startswith("/") or "://" not in source):
            return PyPDFLoader(source)
        if source.startswith("http://") or source.startswith("https://"):
            return WebBaseLoader([source])
        logging.warning(f"No loader available for source: {source}")
        return None

    def build_and_save(self, sources):
        logging.info("Building vector store...")
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
        from langchain_community.vectorstores import FAISS
        self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        self.vector_store.save_local(self.path)
        logging.info(f"Vector store built and saved to {self.path}")
        return True

    def load(self):
        if self.vector_store is not None:
            return True
        if not os.path.exists(self.path):
            logging.warning(f"Vector store path '{self.path}' does not exist.")
            return False
        from langchain_community.vectorstores import FAISS
        self.vector_store = FAISS.load_local(self.path, self.embedding_model, allow_dangerous_deserialization=True)
        logging.info("Vector store loaded from disk.")
        return True
