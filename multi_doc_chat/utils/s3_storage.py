from __future__ import annotations
import os
import boto3
from pathlib import Path
from typing import Optional
from multi_doc_chat.logger import GLOBAL_LOGGER as log


def _s3_client():
    """
    Returns a boto3 S3 client.
    - Locally (LocalStack): AWS_ENDPOINT_URL=http://localstack:4566
    - Cloud (ECS):          AWS_ENDPOINT_URL not set → real AWS
    """
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        endpoint_url=os.getenv("AWS_ENDPOINT_URL"),  # None = real AWS
    )


class S3Storage:
    def __init__(self):
        self.bucket = os.getenv("S3_BUCKET", "multidocchat-local")
        self.client = _s3_client()

    def upload_file(self, local_path: Path, s3_key: str) -> str:
        """Upload a local file to S3. Returns the s3_key."""
        self.client.upload_file(str(local_path), self.bucket, s3_key)
        log.info("Uploaded file to S3", s3_key=s3_key, bucket=self.bucket)
        return s3_key

    def upload_bytes(self, data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to S3. Returns the s3_key."""
        self.client.put_object(
            Body=data,
            Bucket=self.bucket,
            Key=s3_key,
            ContentType=content_type,
        )
        log.info("Uploaded bytes to S3", s3_key=s3_key, size=len(data))
        return s3_key

    def download_file(self, s3_key: str, local_path: Path) -> Path:
        """Download an S3 object to a local path. Returns the local path."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, s3_key, str(local_path))
        log.info("Downloaded file from S3", s3_key=s3_key, local_path=str(local_path))
        return local_path

    def object_exists(self, s3_key: str) -> bool:
        """Return True if the object exists in S3."""
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except self.client.exceptions.ClientError:
            return False

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects whose key starts with prefix. Returns count deleted."""
        paginator = self.client.get_paginator("list_objects_v2")
        deleted = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                self.client.delete_object(Bucket=self.bucket, Key=obj["Key"])
                deleted += 1
        log.info("Deleted S3 objects", prefix=prefix, count=deleted)
        return deleted
