from __future__ import annotations
import os
import uuid
from typing import List

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale

from multi_doc_chat.logger import GLOBAL_LOGGER as log


def _pinecone_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "multidocchat-v2")
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


# ── Legacy dense-only ingest (kept for backward-compatibility) ────────────────

def ingest_documents(docs: List[Document], embeddings, session_id: str) -> int:
    """
    Embed and upsert documents into Pinecone under the given session namespace.
    Returns the number of documents indexed.
    """
    vectorstore = get_pinecone_vectorstore(embeddings, namespace=session_id)
    vectorstore.add_documents(docs)
    log.info("Documents upserted to Pinecone", session_id=session_id, count=len(docs))
    return len(docs)


# ── Hybrid ingest (dense + BM25 sparse + parent_text in metadata) ─────────────

def ingest_documents_hybrid(
    child_docs: List[Document],
    parent_map: dict,
    embeddings,
    session_id: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert child vectors with dense + sparse + parent_text in metadata.

    Args:
        child_docs:  Small child chunks (400 chars) to embed and index.
        parent_map:  Maps parent_id → parent_text (2000-char context).
        embeddings:  Google Gemini embeddings (task_type="retrieval_document").
        session_id:  Pinecone namespace — isolates per-user data.
        batch_size:  Pinecone upsert batch size (max 100 per call).
    """
    encoder = BM25Encoder.default()
    texts = [d.page_content for d in child_docs]

    log.info("Encoding sparse vectors", count=len(texts), session_id=session_id)
    sparse_vecs = encoder.encode_documents(texts)

    log.info("Encoding dense vectors", count=len(texts), session_id=session_id)
    dense_vecs = embeddings.embed_documents(texts)

    index = _pinecone_index()
    batch = []
    for i, (doc, dense, sparse) in enumerate(zip(child_docs, dense_vecs, sparse_vecs)):
        parent_id = doc.metadata.get("parent_id", f"p-{i}")
        batch.append({
            "id": f"{session_id}-child-{i}-{uuid.uuid4().hex[:6]}",
            "values": dense,
            "sparse_values": sparse,
            "metadata": {
                "text": doc.page_content,
                "parent_text": parent_map.get(parent_id, doc.page_content),
                "parent_id": parent_id,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
            },
        })

    for start in range(0, len(batch), batch_size):
        index.upsert(vectors=batch[start : start + batch_size], namespace=session_id)

    log.info(
        "Hybrid vectors upserted",
        count=len(batch),
        session_id=session_id,
    )
    return len(batch)


# ── Hybrid query helper (used by ParentFetchingHybridRetriever) ───────────────

def hybrid_query(
    query: str,
    embeddings,
    session_id: str,
    k: int = 10,
    alpha: float = 0.5,
) -> List[Document]:
    """
    Run a hybrid dense+sparse Pinecone query and return deduplicated parent-level Documents.

    Args:
        query:      User question (already rewritten to standalone form).
        embeddings: Gemini embeddings (task_type="retrieval_query").
        session_id: Pinecone namespace.
        k:          Number of unique parent docs to return.
        alpha:      Blend (0=pure BM25, 1=pure dense).
    """
    encoder = BM25Encoder.default()
    dense_vec = embeddings.embed_query(query)
    sparse_vec = encoder.encode_queries(query)
    hdense, hsparse = hybrid_convex_scale(dense_vec, sparse_vec, alpha=alpha)

    index = _pinecone_index()
    results = index.query(
        vector=hdense,
        sparse_vector=hsparse,
        top_k=k * 2,          # over-fetch — deduplicate by parent_id below
        namespace=session_id,
        include_metadata=True,
    )

    seen: set = set()
    docs: List[Document] = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {})
        pid = meta.get("parent_id", match["id"])
        if pid not in seen:
            seen.add(pid)
            clean_meta = {key: val for key, val in meta.items() if key != "parent_text"}
            docs.append(Document(
                page_content=meta.get("parent_text", meta.get("text", "")),
                metadata=clean_meta,
            ))
        if len(docs) >= k:
            break

    return docs


def delete_session(session_id: str) -> None:
    """Delete all vectors for a session namespace from Pinecone."""
    index = _pinecone_index()
    index.delete(delete_all=True, namespace=session_id)
    log.info("Deleted Pinecone namespace", session_id=session_id)
