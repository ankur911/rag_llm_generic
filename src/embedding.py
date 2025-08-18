from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from src.config import USE_INSTRUCT_E5


def get_e5_explicit_embeddings(
    model_name: str,
    normalize: bool = True,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
):
    """Explicit E5 wrapper: add prefixes yourself; normalize vectors."""
    class _E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"{passage_prefix}{t}" for t in texts])
        def embed_query(self, text):
            return super().embed_query(f"{query_prefix}{text}")

    return _E5Embeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": normalize},
    )

def get_e5_instruct_embeddings(
    model_name: str,
    normalize: bool = True,
    query_prefix: str = "query: ",
    passage_prefix: str = "passage: ",
):
    """Instruct-style variant: uses the LC instruct class; also normalizes."""
    return HuggingFaceInstructEmbeddings(
        model_name=model_name,
        embed_instruction=passage_prefix,
        query_instruction=query_prefix,
        encode_kwargs={"normalize_embeddings": normalize},
    )

def get_embedding_model(
    model_name: str,
    use_instruct: bool = USE_INSTRUCT_E5,
    **kwargs
):
    """Dispatcher used by VectorStoreManager.__init__."""
    if use_instruct:
        return get_e5_instruct_embeddings(model_name, **kwargs)
    return get_e5_explicit_embeddings(model_name, **kwargs)
