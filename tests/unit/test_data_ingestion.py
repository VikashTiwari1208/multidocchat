import pytest
from langchain_core.documents import Document

from multi_doc_chat.src.document_ingestion.data_ingestion import (
    generate_session_id,
    split_documents,
)


def test_generate_session_id_format_and_uniqueness():
    a = generate_session_id()
    b = generate_session_id()
    assert a != b
    assert a.startswith("session_") and b.startswith("session_")
    assert len(a.split("_")) == 4


def test_split_documents_respects_chunk_size():
    docs = [Document(page_content="A" * 1200, metadata={"source": "x.txt"})]
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    assert len(chunks[0].page_content) <= 500


def test_split_documents_preserves_metadata():
    docs = [Document(page_content="Hello world " * 100, metadata={"source": "test.pdf"})]
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert all(c.metadata.get("source") == "test.pdf" for c in chunks)


def test_split_documents_empty_input():
    chunks = split_documents([])
    assert chunks == []
