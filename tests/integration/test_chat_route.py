import pytest


def test_chat_invalid_session_returns_400(client, stub_dynamo, stub_rag):
    resp = client.post("/chat", json={"session_id": "nope", "message": "hi"})
    assert resp.status_code == 400
    assert "Invalid or expired" in resp.json()["detail"]


def test_chat_empty_message_returns_400(client, stub_dynamo, stub_rag):
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    DynamoSessionStore().create_session("sess_test", "s3://fake/key", filename="test.txt")

    resp = client.post("/chat", json={"session_id": "sess_test", "message": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_chat_not_ready_returns_400(client, stub_dynamo, stub_rag):
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    dynamo = DynamoSessionStore()
    dynamo.create_session("sess_processing", "s3://fake/key", filename="test.txt")
    dynamo.set_status("sess_processing", "processing")

    resp = client.post("/chat", json={"session_id": "sess_processing", "message": "hello"})
    assert resp.status_code == 400
    assert "indexed" in resp.json()["detail"].lower()


def test_chat_success_returns_answer(client, stub_dynamo, stub_rag):
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    DynamoSessionStore().create_session("sess_ready", "s3://fake/key", filename="test.txt")

    resp = client.post("/chat", json={"session_id": "sess_ready", "message": "What is RAG?"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "stubbed answer"


def test_chat_saves_history(client, stub_dynamo, stub_rag):
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    dynamo = DynamoSessionStore()
    dynamo.create_session("sess_history", "s3://fake/key", filename="test.txt")

    client.post("/chat", json={"session_id": "sess_history", "message": "Hello"})
    history = dynamo.get_history("sess_history")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_sessions_list_returns_active(client, stub_dynamo):
    from multi_doc_chat.utils.dynamo_store import DynamoSessionStore
    DynamoSessionStore().create_session("sess_list", "s3://fake/key", filename="report.pdf")

    resp = client.get("/sessions")
    assert resp.status_code == 200
    sessions = resp.json()["sessions"]
    assert any(s["session_id"] == "sess_list" for s in sessions)
