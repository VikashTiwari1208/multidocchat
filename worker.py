"""
SQS Worker — async ingestion pipeline.

Responsibilities:
  1. Poll SQS for ingestion jobs queued by the API on upload
  2. Download the raw document from S3 to a temp dir
  3. Extract text, chunk, embed via Google AI
  4. Upsert vectors into Pinecone (session_id as namespace)
  5. Mark session status = "ready" in DynamoDB
  6. Delete the SQS message on success (SQS retries on failure automatically)

Environment variables required:
  SQS_QUEUE_URL     — full SQS queue URL
  S3_BUCKET         — S3 bucket name
  DYNAMO_TABLE      — DynamoDB table name (default: multidocchat-sessions)
  PINECONE_API_KEY  — Pinecone API key
  PINECONE_INDEX    — Pinecone index name (default: multidocchat-v2)
  GROQ_API_KEY      — required by ModelLoader
  AWS_ENDPOINT_URL  — (local only) LocalStack endpoint
"""
from __future__ import annotations
import json
import os
import sys
import time
import tempfile
from pathlib import Path

import boto3
import yaml
from dotenv import load_dotenv

# Load .env in local dev; in ECS secrets come from Secrets Manager env vars
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from multi_doc_chat.utils.s3_storage import S3Storage
from multi_doc_chat.utils.dynamo_store import DynamoSessionStore, STATUS_READY, STATUS_FAILED
from multi_doc_chat.utils.pinecone_store import ingest_documents_hybrid
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.document_ops import load_documents
from multi_doc_chat.logger import GLOBAL_LOGGER as log

_CONFIG_PATH = Path(__file__).parent / "multi_doc_chat" / "config" / "config.yaml"
_cfg = yaml.safe_load(_CONFIG_PATH.read_text())


# SQS visibility timeout must be >= max expected processing time (seconds)
VISIBILITY_TIMEOUT = int(os.getenv("SQS_VISIBILITY_TIMEOUT", "3600"))  # 1 hour — large docs on free-tier Gemini need time for rate-limit retries
POLL_WAIT_SECONDS = 20   # SQS long polling — reduces empty receive API calls


def _sqs_client():
    return boto3.client(
        "sqs",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
    )


def process_job(body: dict) -> None:
    """
    Full ingestion pipeline for one document:
      S3 download → load + chunk → embed → Pinecone upsert → DynamoDB ready
    """
    session_id: str = body["session_id"]
    s3_key: str = body["s3_key"]
    filename: str = body["filename"]

    s3 = S3Storage()
    dynamo = DynamoSessionStore()

    log.info("Job started", session_id=session_id, s3_key=s3_key)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download file from S3
        local_path = Path(tmpdir) / filename
        s3.download_file(s3_key, local_path)

        # 2. Load docs (PyPDF with OCR fallback for scanned PDFs)
        docs = load_documents([local_path])
        if not docs:
            raise ValueError(f"No documents extracted from {filename}")

        text_docs = [d for d in docs if d.page_content and d.page_content.strip()]
        log.info("Documents loaded", total_pages=len(docs), text_pages=len(text_docs), session_id=session_id)

        if not text_docs:
            raise ValueError(
                f"All {len(docs)} pages in '{filename}' appear empty after text extraction and OCR. "
                "Upload a PDF with selectable text or a higher-quality scan."
            )

        # 3. Two-level split: large parent chunks for context, small child chunks for retrieval
        c = _cfg["chunking"]
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c["parent_chunk_size"], chunk_overlap=c["parent_chunk_overlap"]
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=c["child_chunk_size"], chunk_overlap=c["child_chunk_overlap"]
        )

        parent_chunks = parent_splitter.split_documents(text_docs)
        child_docs, parent_map = [], {}
        for p_idx, parent in enumerate(parent_chunks):
            parent_id = f"parent-{session_id}-{p_idx}"
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = parent_id
                child_docs.append(child)
            parent_map[parent_id] = parent.page_content

        log.info(
            "Two-level split complete",
            parents=len(parent_chunks),
            children=len(child_docs),
            session_id=session_id,
        )

        if not child_docs:
            raise ValueError(f"Splitting produced 0 child chunks for '{filename}'")

        # 4. Embed (dense + BM25 sparse) and upsert hybrid vectors to Pinecone
        embeddings = ModelLoader().load_embeddings(task_type="retrieval_document")
        indexed = ingest_documents_hybrid(child_docs, parent_map, embeddings, session_id=session_id)
        log.info("Hybrid vectors upserted to Pinecone", indexed=indexed, session_id=session_id)

    # 4. Mark session ready — outside tempdir (file already processed)
    dynamo.set_status(session_id, STATUS_READY)
    log.info("Job completed", session_id=session_id)


def run() -> None:
    sqs = _sqs_client()
    queue_url = os.getenv("SQS_QUEUE_URL")
    if not queue_url:
        log.error("SQS_QUEUE_URL not set — worker cannot start")
        sys.exit(1)

    log.info("Worker started", queue_url=queue_url)

    while True:
        try:
            resp = sqs.receive_message(
                QueueUrl=queue_url,
                MaxNumberOfMessages=1,        # process one at a time (safe for large docs)
                WaitTimeSeconds=POLL_WAIT_SECONDS,
                VisibilityTimeout=VISIBILITY_TIMEOUT,
            )

            messages = resp.get("Messages", [])
            if not messages:
                continue  # long poll returned empty — loop again

            msg = messages[0]
            receipt = msg["ReceiptHandle"]

            try:
                body = json.loads(msg["Body"])
                process_job(body)

                # Delete on success — prevents re-processing
                sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
                log.info("SQS message deleted after successful processing")

            except Exception as e:
                log.error("Job failed — message will retry after visibility timeout", error=str(e))

                # Mark session as failed so frontend stops polling
                try:
                    session_id = json.loads(msg["Body"]).get("session_id", "unknown")
                    DynamoSessionStore().set_status(session_id, STATUS_FAILED)
                except Exception:
                    pass

        except KeyboardInterrupt:
            log.info("Worker shutting down")
            break
        except Exception as e:
            log.error("SQS polling error — retrying in 5s", error=str(e))
            time.sleep(5)


if __name__ == "__main__":
    run()
