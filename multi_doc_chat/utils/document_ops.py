from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from fastapi import UploadFile

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _ocr_pdf(path: Path) -> List[Document]:
    """
    OCR fallback for scanned/image-based PDFs.
    Converts each page to an image, runs Tesseract, returns one Document per page.
    Requires: pdf2image (poppler-utils) + pytesseract (tesseract-ocr) installed.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        log.warning("OCR dependencies not available (pdf2image / pytesseract) — skipping OCR")
        return []

    log.info("Falling back to OCR for scanned PDF", path=str(path))
    images = convert_from_path(str(path), dpi=200)
    ocr_docs: List[Document] = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            ocr_docs.append(Document(
                page_content=text,
                metadata={"source": str(path), "page": i},
            ))
    log.info("OCR completed", pages_with_text=len(ocr_docs), total_pages=len(images))
    return ocr_docs


def load_documents(paths: Iterable[Path]) -> List[Document]:
    """
    Load docs using the appropriate loader based on extension.
    For PDFs: tries PyPDFLoader first; if all pages are empty (scanned PDF),
    falls back to Tesseract OCR via pdf2image.
    """
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                pdf_docs = PyPDFLoader(str(p)).load()
                # Check if text was actually extracted
                has_text = any(d.page_content and d.page_content.strip() for d in pdf_docs)
                if not has_text:
                    log.warning("PyPDF found no text — attempting OCR", path=str(p), pages=len(pdf_docs))
                    pdf_docs = _ocr_pdf(p)
                docs.extend(pdf_docs)
            elif ext == ".docx":
                docs.extend(Docx2txtLoader(str(p)).load())
            elif ext == ".txt":
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue
        log.info("Documents loaded", count=len(docs))
        return docs
    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException("Error loading documents", e) from e
    

class FastAPIFileAdapter:
    """Adapt FastAPI UploadFile to a simple object with .name and .getbuffer()."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()