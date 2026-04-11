from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.utils.s3_storage import S3Storage
from multi_doc_chat.utils.dynamo_store import DynamoSessionStore, STATUS_READY
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.src.document_ingestion.data_ingestion import generate_session_id
from multi_doc_chat.logger import GLOBAL_LOGGER as log

# ------------------------------------
# Module-level embedding model cache
# Loaded once on first use, reused for all requests — avoids reloading 440MB model
# ------------------------------------
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        log.info("Loading embedding model into cache (first request)")
        _embeddings = ModelLoader().load_embeddings()
    return _embeddings


# ----------------------------
# FastAPI initialization
# ----------------------------
app = FastAPI(
    title="MultiDocChat by Vikash",
    description="A production-grade multi-document RAG chatbot — upload PDFs, DOCX, or TXT files and chat with them using LangChain + Pinecone + Groq. Deployed on AWS ECS Fargate.",
    version="1.0.0",
    contact={"name": "Vikash", "url": "https://github.com/VikashTiwari1208"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ----------------------------
# SQS helper
# ----------------------------
def _sqs_client():
    return boto3.client(
        "sqs",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),  # None = real AWS
    )


def _enqueue_job(session_id: str, s3_key: str, filename: str) -> None:
    sqs = _sqs_client()
    sqs.send_message(
        QueueUrl=os.getenv("SQS_QUEUE_URL"),
        MessageBody=json.dumps({
            "session_id": session_id,
            "s3_key": s3_key,
            "filename": filename,
        }),
    )
    log.info("Job enqueued to SQS", session_id=session_id, s3_key=s3_key)


# ----------------------------
# Models
# ----------------------------
class UploadResponse(BaseModel):
    session_id: str
    message: str


class StatusResponse(BaseModel):
    session_id: str
    status: str          # processing | ready | failed


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    answer: str


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)) -> UploadResponse:
    """
    Upload a document, save it to S3, and queue an ingestion job.
    Returns immediately — the document is indexed asynchronously by the worker.
    Poll GET /status/{session_id} to know when it is ready.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    file = files[0]  # one document per session
    filename = file.filename or "upload"
    session_id = generate_session_id()
    s3_key = f"uploads/{session_id}/{filename}"

    try:
        # Save to temp file then upload to S3
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        S3Storage().upload_file(tmp_path, s3_key)
        tmp_path.unlink(missing_ok=True)

        # Create session in DynamoDB with status=processing
        DynamoSessionStore().create_session(session_id, s3_key, filename=filename)

        # Enqueue ingestion job for the worker
        _enqueue_job(session_id, s3_key, filename)

        log.info("Upload accepted", session_id=session_id, filename=filename)
        return UploadResponse(
            session_id=session_id,
            message="Document uploaded. Indexing in progress — poll /status/{session_id}.",
        )

    except Exception as e:
        log.error("Upload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


@app.get("/sessions")
def list_sessions():
    """Return all active (non-expired) sessions for the document switcher."""
    return {"sessions": DynamoSessionStore().list_sessions()}


@app.get("/status/{session_id}", response_model=StatusResponse)
def get_status(session_id: str) -> StatusResponse:
    """
    Poll this endpoint after upload to check if the document has been indexed.
    Returns: processing | ready | failed
    """
    status = DynamoSessionStore().get_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Session not found. Re-upload the document.")
    return StatusResponse(session_id=session_id, status=status)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session_id = req.session_id
    message = req.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    dynamo = DynamoSessionStore()

    if not dynamo.session_exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid or expired session. Re-upload documents.")

    if not dynamo.is_ready(session_id):
        raise HTTPException(status_code=400, detail="Document still being indexed. Please wait.")

    try:
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_pinecone(
            session_id=session_id,
            search_type="mmr",
            fetch_k=20,
            lambda_mult=0.5,
            embeddings=get_embeddings(),   # reuse cached model
        )

        simple = dynamo.get_history(session_id)
        lc_history = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in simple
        ]

        answer = rag.invoke(message, chat_history=lc_history)
        simple.append({"role": "user",      "content": message})
        simple.append({"role": "assistant", "content": answer})
        dynamo.save_history(session_id, simple)
        return ChatResponse(answer=answer)

    except DocumentPortalException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        log.error("Chat failed", error=str(e), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming variant of /chat — returns Server-Sent Events.
    Text chunks arrive word-by-word as the LLM generates them.
    Use this from Streamlit with st.write_stream().
    """
    session_id = req.session_id
    message = req.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    dynamo = DynamoSessionStore()
    if not dynamo.session_exists(session_id):
        raise HTTPException(status_code=400, detail="Invalid or expired session.")
    if not dynamo.is_ready(session_id):
        raise HTTPException(status_code=400, detail="Document still being indexed.")

    rag = ConversationalRAG(session_id=session_id)
    rag.load_retriever_from_pinecone(
        session_id=session_id,
        search_type="mmr",
        fetch_k=20,
        lambda_mult=0.5,
        embeddings=get_embeddings(),
    )

    simple = dynamo.get_history(session_id)
    lc_history = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in simple
    ]

    async def generate():
        collected = []
        try:
            async for chunk in rag.chain.astream({"input": message, "chat_history": lc_history}):
                collected.append(chunk)
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # Persist full answer after stream completes
            full = "".join(collected)
            simple.append({"role": "user",      "content": message})
            simple.append({"role": "assistant",  "content": full})
            dynamo.save_history(session_id, simple)
            log.info("Stream completed", session_id=session_id, chars=len(full))
        except Exception as e:
            log.error("Stream error", error=str(e), session_id=session_id)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
