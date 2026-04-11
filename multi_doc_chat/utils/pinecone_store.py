from __future__ import annotations
import os
from typing import List
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from multi_doc_chat.logger import GLOBAL_LOGGER as log


def _pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "multidocchat")
    if not api_key:
        raise ValueError("PINECONE_API_KEY env var not set")
    pc = Pinecone(api_key=api_key)
    return pc.Index(index_name)


def get_pinecone_vectorstore(embeddings, namespace: str) -> PineconeVectorStore:
    """Return a PineconeVectorStore scoped to a session namespace."""
    return PineconeVectorStore(
        index=_pinecone_index(),
        embedding=embeddings,
        namespace=namespace,
    )


def ingest_documents(docs: List[Document], embeddings, session_id: str) -> int:
    """
    Embed and upsert documents into Pinecone under the given session namespace.
    Returns the number of documents indexed.
    """
    vectorstore = get_pinecone_vectorstore(embeddings, namespace=session_id)
    vectorstore.add_documents(docs)
    log.info("Documents upserted to Pinecone", session_id=session_id, count=len(docs))
    return len(docs)


def delete_session(session_id: str) -> None:
    """Delete all vectors for a session namespace from Pinecone."""
    index = _pinecone_index()
    index.delete(delete_all=True, namespace=session_id)
    log.info("Deleted Pinecone namespace", session_id=session_id)
