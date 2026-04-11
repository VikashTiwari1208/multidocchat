import io
import pytest


def test_upload_returns_session_id(client, stub_s3, stub_dynamo, stub_sqs):
    files = {"files": ("note.txt", io.BytesIO(b"hello world"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["session_id"].startswith("session_")


def test_upload_no_files_returns_422(client, stub_s3, stub_dynamo, stub_sqs):
    resp = client.post("/upload", files=[])
    assert resp.status_code == 422


def test_upload_creates_dynamo_session(client, stub_s3, stub_dynamo, stub_sqs):
    files = {"files": ("doc.txt", io.BytesIO(b"content"), "text/plain")}
    resp = client.post("/upload", files=files)
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    # Session should exist in DynamoDB after upload
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    assert DynamoSessionStore().session_exists(session_id)


def test_status_returns_processing_or_ready(client, stub_s3, stub_dynamo, stub_sqs):
    files = {"files": ("doc.txt", io.BytesIO(b"content"), "text/plain")}
    upload_resp = client.post("/upload", files=files)
    session_id = upload_resp.json()["session_id"]

    status_resp = client.get(f"/status/{session_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] in ("processing", "ready", "failed")


def test_status_unknown_session_returns_404(client, stub_dynamo):
    resp = client.get("/status/nonexistent-session")
    assert resp.status_code == 404
