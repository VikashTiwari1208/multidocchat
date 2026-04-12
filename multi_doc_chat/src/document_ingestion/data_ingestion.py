from __future__ import annotations
from datetime import datetime
from typing import List
import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.logger import GLOBAL_LOGGER as log


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def generate_session_id() -> str:
    """Generate a unique session ID with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


def split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
    return chunks
