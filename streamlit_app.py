"""
Streamlit frontend for MultiDocChat.
Connects to the FastAPI backend (API_URL env var, default http://app:8080).

Features:
  - Document switcher — all previously uploaded (non-expired) docs listed
  - File upload with real-time indexing progress (polls /status/{session_id})
  - Streaming chat responses via /chat/stream (text appears word-by-word)
  - Full conversation history per session
"""
from __future__ import annotations
import json
import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://app:8080")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MultiDocChat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [
    ("session_id",  None),
    ("messages",    []),
    ("indexed",     False),
    ("filename",    None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helper ───────────────────────────────────────────────────────────────────
def fetch_sessions() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/sessions", timeout=5)
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception:
        return []


def switch_to(session: dict):
    st.session_state.session_id = session["session_id"]
    st.session_state.filename   = session["filename"]
    st.session_state.indexed    = session["status"] == "ready"
    st.session_state.messages   = []   # history lives in DynamoDB; reload on next chat


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 MultiDocChat")
    st.caption("Upload · Index · Chat — powered by RAG")
    st.divider()

    # ── Upload new document ───────────────────────────────────────────────────
    with st.expander("➕ Upload new document", expanded=not st.session_state.indexed):
        uploaded = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            help="PDF, Word, or plain text",
            label_visibility="collapsed",
        )
        if uploaded and st.button("🚀 Upload & Index", use_container_width=True, type="primary"):
            # 1. Upload to FastAPI → S3 → SQS
            with st.spinner("Uploading…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/upload",
                        files={"files": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.session_id = data["session_id"]
                    st.session_state.indexed    = False
                    st.session_state.messages   = []
                    st.session_state.filename   = uploaded.name
                except Exception as e:
                    st.error(f"Upload failed: {e}")
                    st.stop()

            # 2. Poll /status until ready or failed
            progress = st.progress(0)
            tick = 0
            while True:
                try:
                    sr = requests.get(
                        f"{API_URL}/status/{st.session_state.session_id}", timeout=10
                    )
                    status = sr.json().get("status", "processing")
                except Exception:
                    status = "processing"

                tick += 1
                progress.progress(min(tick * 10, 90), text=f"Indexing… ({tick * 2}s elapsed)")

                if status == "ready":
                    progress.progress(100, text="✅ Ready!")
                    st.session_state.indexed = True
                    time.sleep(0.5)
                    st.rerun()
                elif status == "failed":
                    progress.empty()
                    st.error("❌ Indexing failed — please try re-uploading.")
                    st.stop()

                time.sleep(2)

    st.divider()

    # ── Document switcher ─────────────────────────────────────────────────────
    st.markdown("**📂 Your documents**")

    sessions = fetch_sessions()
    ready_sessions = [s for s in sessions if s["status"] == "ready"]

    if not ready_sessions:
        st.caption("No documents indexed yet. Upload one above.")
    else:
        for s in ready_sessions:
            is_active = s["session_id"] == st.session_state.session_id
            label = f"{'▶ ' if is_active else ''}{s['filename']}"
            # Show creation time nicely
            created = s.get("created_at", "")[:16].replace("T", " ") if s.get("created_at") else ""
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{label}**" if is_active else label,
                    help=f"Uploaded: {created}" if created else None,
                )
            with col2:
                if not is_active:
                    if st.button("Switch", key=f"sw_{s['session_id']}", use_container_width=True):
                        switch_to(s)
                        st.rerun()
                else:
                    st.caption("active")

    st.divider()
    st.caption("Built by **Vikash** · FastAPI · LangChain · Pinecone · AWS ECS")


# ── Main area — chat ──────────────────────────────────────────────────────────
if not st.session_state.indexed:
    st.markdown("## 👋 Welcome to MultiDocChat")
    st.info(
        "Upload a PDF, DOCX, or TXT file in the sidebar to get started.\n\n"
        "Already uploaded something? Select it from **Your documents** in the sidebar."
    )
    st.stop()

st.markdown(f"### 💬 Chatting with *{st.session_state.filename}*")
st.divider()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about your document…"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        def _stream():
            """Generator that reads SSE chunks from /chat/stream."""
            with requests.post(
                f"{API_URL}/chat/stream",
                json={"session_id": st.session_state.session_id, "message": prompt},
                stream=True,
                timeout=120,
            ) as r:
                r.raise_for_status()
                for raw in r.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                        if "error" in data:
                            yield f"\n\n⚠️ {data['error']}"
                            break
                        yield data.get("chunk", "")
                    except json.JSONDecodeError:
                        continue

        try:
            full_answer = st.write_stream(_stream())
        except Exception as e:
            full_answer = f"⚠️ Error: {e}"
            st.error(full_answer)

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
