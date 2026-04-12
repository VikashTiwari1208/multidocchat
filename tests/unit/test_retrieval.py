import pytest

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.exception.custom_exception import DocumentPortalException


def test_conversationalrag_raises_when_chain_not_initialized(stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(DocumentPortalException):
        rag.invoke("hello")
