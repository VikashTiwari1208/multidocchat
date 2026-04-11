import os
import pathlib
import sys
import pytest

os.environ.setdefault("PYTHONPATH", str(pathlib.Path(__file__).resolve().parents[1] / "multi_doc_chat"))
os.environ.setdefault("GROQ_API_KEY", "dummy")
os.environ.setdefault("GOOGLE_API_KEY", "dummy")
os.environ.setdefault("LLM_PROVIDER", "groq")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MULTI_DOC = ROOT / "multi_doc_chat"
for p in (str(ROOT), str(MULTI_DOC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import main
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def tmp_dirs(tmp_path: pathlib.Path):
    data_dir = tmp_path / "data"
    faiss_dir = tmp_path / "faiss_index"
    data_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    cwd = pathlib.Path.cwd()
    try:
        os.chdir(tmp_path)
        yield {"data": data_dir, "faiss": faiss_dir}
    finally:
        os.chdir(cwd)


class _StubEmbeddings:
    def embed_query(self, text):
        return [0.0, 0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.0, 0.1, 0.2] for _ in texts]

    def __call__(self, text):
        return [0.0, 0.1, 0.2]


class _StubLLM:
    def invoke(self, input):
        return "stubbed answer"


@pytest.fixture
def stub_model_loader(monkeypatch):
    import utils.model_loader as ml_mod
    from multi_doc_chat.utils import model_loader as ml_mod2

    class FakeApiKeyMgr:
        def __init__(self):
            self.api_keys = {"GROQ_API_KEY": "x", "GOOGLE_API_KEY": "y"}

        def get(self, key):
            return self.api_keys.get(key, "dummy")

    class FakeModelLoader:
        def __init__(self):
            self.api_key_mgr = FakeApiKeyMgr()
            self.config = {
                "embedding_model": {"provider": "huggingface", "model_name": "fake-embed"},
                "llm": {
                    "groq": {
                        "provider": "groq",
                        "model_name": "fake-llm",
                        "temperature": 0.0,
                        "max_output_tokens": 128,
                    }
                },
            }

        def load_embeddings(self):
            return _StubEmbeddings()

        def load_llm(self):
            return _StubLLM()

    monkeypatch.setattr(ml_mod, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(ml_mod2, "ApiKeyManager", FakeApiKeyMgr)
    monkeypatch.setattr(ml_mod2, "ModelLoader", FakeModelLoader)

    import multi_doc_chat.src.document_ingestion.data_ingestion as di
    import multi_doc_chat.src.document_chat.retrieval as r
    monkeypatch.setattr(di, "ModelLoader", FakeModelLoader)
    monkeypatch.setattr(r, "ModelLoader", FakeModelLoader)
    yield FakeModelLoader


@pytest.fixture
def stub_dynamo(monkeypatch):
    """Replace DynamoSessionStore with an in-memory fake — no real DynamoDB needed."""
    from multi_doc_chat.utils import dynamo_store

    class FakeDynamo:
        _store: dict = {}

        def __init__(self):
            pass

        def create_session(self, session_id, s3_key, filename=""):
            FakeDynamo._store[session_id] = {
                "status": "ready", "history": [], "filename": filename
            }

        def set_status(self, session_id, status):
            if session_id in FakeDynamo._store:
                FakeDynamo._store[session_id]["status"] = status

        def get_status(self, session_id):
            return FakeDynamo._store.get(session_id, {}).get("status")

        def session_exists(self, session_id):
            return session_id in FakeDynamo._store

        def is_ready(self, session_id):
            return FakeDynamo._store.get(session_id, {}).get("status") == "ready"

        def get_history(self, session_id):
            return FakeDynamo._store.get(session_id, {}).get("history", [])

        def save_history(self, session_id, history):
            if session_id in FakeDynamo._store:
                FakeDynamo._store[session_id]["history"] = history

        def list_sessions(self):
            return [
                {"session_id": k, "filename": v["filename"],
                 "status": v["status"], "created_at": ""}
                for k, v in FakeDynamo._store.items()
            ]

    FakeDynamo._store.clear()
    monkeypatch.setattr(dynamo_store, "DynamoSessionStore", FakeDynamo)
    monkeypatch.setattr(main, "DynamoSessionStore", FakeDynamo)
    yield FakeDynamo


@pytest.fixture
def stub_s3(monkeypatch):
    """Replace S3Storage with a no-op fake — no real S3 needed."""
    from multi_doc_chat.utils import s3_storage

    class FakeS3:
        def __init__(self):
            pass

        def upload_file(self, _local_path, s3_key):
            return f"s3://fake/{s3_key}"

    monkeypatch.setattr(s3_storage, "S3Storage", FakeS3)
    monkeypatch.setattr(main, "S3Storage", FakeS3)
    yield FakeS3


@pytest.fixture
def stub_sqs(monkeypatch):
    """Stub out SQS enqueue so tests don't hit real AWS."""
    monkeypatch.setattr(main, "_enqueue_job", lambda *a, **kw: None)


@pytest.fixture
def stub_rag(monkeypatch):
    import multi_doc_chat.src.document_chat.retrieval as r

    class FakeRAG:
        def __init__(self, session_id=None, retriever=None):
            self.session_id = session_id
            self.chain = None

        def load_retriever_from_pinecone(self, session_id, embeddings=None, **kwargs):
            return None

        def load_retriever_from_faiss(self, index_path, **kwargs):
            return None

        def invoke(self, user_input, chat_history=None):
            return "stubbed answer"

    monkeypatch.setattr(r, "ConversationalRAG", FakeRAG)
    monkeypatch.setattr(main, "ConversationalRAG", FakeRAG)
    yield FakeRAG
