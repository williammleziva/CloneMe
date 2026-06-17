from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
QDRANT_PATH = "./data/qdrant_local"
COLLECTION_NAME = "facts"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def get_vectorstore(embeddings: HuggingFaceEmbeddings | None = None) -> QdrantVectorStore:
    if embeddings is None:
        embeddings = get_embeddings()

    client = get_client()
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def get_retriever(k: int = 5):
    return get_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
