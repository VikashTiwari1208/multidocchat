from __future__ import annotations
import json
import os
import boto3
from datetime import datetime, timezone
from typing import List, Optional
from multi_doc_chat.logger import GLOBAL_LOGGER as log


def _dynamo_client():
    """
    Returns a boto3 DynamoDB client.
    - Locally (LocalStack): AWS_ENDPOINT_URL=http://localstack:4566
    - Cloud (ECS):          AWS_ENDPOINT_URL not set → real AWS
    """
    return boto3.client(
        "dynamodb",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),  # None = real AWS
    )


# Session lifecycle statuses
STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"

# How long a session lives in DynamoDB (seconds)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))  # 1 hour default


class DynamoSessionStore:
    """
    Stores session status and chat history in DynamoDB.

    Table schema (partition key: session_id):
        session_id  S   — unique session identifier
        status      S   — processing | ready | failed
        s3_key      S   — S3 key of the uploaded document
        history     S   — JSON-encoded list of {role, content} dicts
        ttl         N   — Unix timestamp for DynamoDB TTL auto-expiry
    """

    def __init__(self):
        self.client = _dynamo_client()
        self.table = os.getenv("DYNAMO_TABLE", "multidocchat-sessions")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, session_id: str, s3_key: str, filename: str = "") -> None:
        """Create a new session with status=processing."""
        ttl = int(datetime.now(timezone.utc).timestamp()) + SESSION_TTL_SECONDS
        created_at = datetime.now(timezone.utc).isoformat()
        self.client.put_item(
            TableName=self.table,
            Item={
                "session_id": {"S": session_id},
                "status":     {"S": STATUS_PROCESSING},
                "s3_key":     {"S": s3_key},
                "filename":   {"S": filename},
                "created_at": {"S": created_at},
                "history":    {"S": "[]"},
                "ttl":        {"N": str(ttl)},
            },
        )
        log.info("Session created in DynamoDB", session_id=session_id, s3_key=s3_key, filename=filename)

    def set_status(self, session_id: str, status: str) -> None:
        self.client.update_item(
            TableName=self.table,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET #st = :s",
            ExpressionAttributeNames={"#st": "status"},
            ExpressionAttributeValues={":s": {"S": status}},
        )
        log.info("Session status updated", session_id=session_id, status=status)

    def get_status(self, session_id: str) -> Optional[str]:
        """Returns status string or None if session does not exist."""
        resp = self.client.get_item(
            TableName=self.table,
            Key={"session_id": {"S": session_id}},
            ProjectionExpression="#st",
            ExpressionAttributeNames={"#st": "status"},
        )
        item = resp.get("Item")
        return item["status"]["S"] if item else None

    def session_exists(self, session_id: str) -> bool:
        return self.get_status(session_id) is not None

    def is_ready(self, session_id: str) -> bool:
        return self.get_status(session_id) == STATUS_READY

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> List[dict]:
        resp = self.client.get_item(
            TableName=self.table,
            Key={"session_id": {"S": session_id}},
            ProjectionExpression="history",
        )
        item = resp.get("Item")
        if not item:
            return []
        return json.loads(item.get("history", {}).get("S", "[]"))

    def save_history(self, session_id: str, history: List[dict]) -> None:
        self.client.update_item(
            TableName=self.table,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET history = :h",
            ExpressionAttributeValues={":h": {"S": json.dumps(history)}},
        )

    # ------------------------------------------------------------------
    # Session listing (for document switcher in UI)
    # ------------------------------------------------------------------

    def list_sessions(self) -> List[dict]:
        """
        Returns all non-expired sessions sorted newest first.
        Each entry: {session_id, filename, status, created_at}
        """
        now = int(datetime.now(timezone.utc).timestamp())
        resp = self.client.scan(
            TableName=self.table,
            ProjectionExpression="session_id, filename, #st, created_at, #ttl",
            ExpressionAttributeNames={"#st": "status", "#ttl": "ttl"},
            FilterExpression="#ttl > :now",
            ExpressionAttributeValues={":now": {"N": str(now)}},
        )
        sessions = []
        for item in resp.get("Items", []):
            sessions.append({
                "session_id": item["session_id"]["S"],
                "filename":   item.get("filename", {}).get("S", "Unknown"),
                "status":     item.get("status",   {}).get("S", "unknown"),
                "created_at": item.get("created_at", {}).get("S", ""),
            })
        # Sort newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions
