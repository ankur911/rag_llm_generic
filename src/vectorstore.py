"""
Vector store management for RAG pipeline.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import requests
import trafilatura
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from src.config import VECTOR_STORE_PATH, EMBEDDING_MODEL, USE_INSTRUCT_E5, CHUNK_SIZE, CHUNK_OVERLAP
from src.embedding import get_embedding_model

class VectorStoreManager:
    def __init__(self, path: str = VECTOR_STORE_PATH, embedding_model=None, model_name: str | None = None):
        self.path = path
        default_model_name = model_name or EMBEDDING_MODEL
        # choose implementation via config flag
        self.embedding_model = embedding_model or get_embedding_model(
            default_model_name,
            use_instruct=USE_INSTRUCT_E5,
        )
        self.vector_store = None

    # ---- Clean HTML fetch using Trafilatura ----
    def _fetch_clean_html(self, url: str) -> str:
        try:
            resp = requests.get(
                url,
                timeout=20,
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; MyRAGBot/1.0; +https://example.com/contact)"
                } # Many sites block or throttle the default python-requests user agent; a browser-like UA avoids 403/429s
                #It helps the server identify what’s making the request (good practice vs. a blank UA).
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            logging.error(f"HTTP fetch failed for {url}: {e}")
            return ""
        try:
            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
            )
            return text or ""
        except Exception as e:
            logging.error(f"Trafilatura extract failed for {url}: {e}")
            return ""

    def _get_loader(self, source: str) -> Optional[Union[PyPDFLoader, List[Document]]]:
        # Local PDF (supports Unix absolute or Windows drive paths; exclude URLs)
        if source.lower().endswith(".pdf") and "://" not in source:
            return PyPDFLoader(source)

        # Web URL
        if source.startswith(("http://", "https://")):
            text = self._fetch_clean_html(source)
            if not text or not text.strip():
                logging.warning(f"No extractable text from {source}")
                return None
            # Return a list of Document objects as a loader shim
            return [Document(page_content=text.strip(), metadata={"source": source})]

        logging.warning(f"No loader available for source: {source}")
        return None

    def build_and_save(self, sources: List[str]) -> bool:
        logging.info("Building vector store...")
        docs: List[Document] = []

        for src in sources:
            loader = self._get_loader(src)
            if not loader:
                continue
            try:
                if isinstance(loader, list):  # our shimmed web loader
                    loaded = loader
                else:  # PyPDFLoader (or other real loader)
                    loaded = loader.load()
                docs.extend(loaded)
                logging.info(f"Loaded {len(loaded)} docs from {src}")
            except Exception as e:
                logging.error(f"Failed to load source {src}: {e}", exc_info=True)

        if not docs:
            logging.error("No documents were loaded. Vector store not built.")
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        logging.info(f"Split {len(docs)} raw docs into {len(chunks)} chunks.")

        from langchain_community.vectorstores import FAISS

        self.vector_store = FAISS.from_documents(chunks, self.embedding_model)
        os.makedirs(self.path, exist_ok=True)
        self.vector_store.save_local(self.path)

        # Write simple metadata to help detect mismatches later
        meta = {
            "embedding_model": getattr(self.embedding_model, "model_name", "unknown"),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "sources": sources,
        }
        Path(self.path, "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        logging.info(f"Vector store built and saved to {self.path}")
        return True

    def load(self) -> bool:
        if self.vector_store is not None:
            return True

        if not os.path.exists(self.path):
            logging.warning(f"Vector store path '{self.path}' does not exist.")
            return False

        from langchain_community.vectorstores import FAISS

        self.vector_store = FAISS.load_local(
            self.path,
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

        # Optional: warn if metadata mismatches current config
        try:
            meta_path = Path(self.path, "meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                current_model = getattr(self.embedding_model, "model_name", "unknown")
                if (
                    meta.get("embedding_model") != current_model
                    or meta.get("chunk_size") != CHUNK_SIZE
                    or meta.get("chunk_overlap") != CHUNK_OVERLAP
                ):
                    logging.warning(
                        "Vector store metadata differs from current config "
                        f"(stored: {meta}, current: "
                        f"{{'embedding_model': '{current_model}', "
                        f"'chunk_size': {CHUNK_SIZE}, 'chunk_overlap': {CHUNK_OVERLAP}}}). "
                        "Consider rebuilding."
                    )
        except Exception as e:
            logging.debug(f"Could not read/compare meta.json: {e}")

        logging.info("Vector store loaded from disk.")
        return True
