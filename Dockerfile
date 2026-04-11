# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir
WORKDIR /app

# Install OS dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential poppler-utils tesseract-ocr curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV PYTHONPATH="/app:/app/multi_doc_chat"

# Copy dependency manifests for better layer caching
COPY requirements.txt ./

# Install CPU-only PyTorch BEFORE requirements.txt so sentence-transformers
# doesn't pull the default CUDA-enabled wheel (~2 GB of unused GPU libraries)
RUN uv pip install --system torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN uv pip install --system -r requirements.txt && \
    uv pip uninstall --system pinecone-plugin-inference pinecone-plugin-assistant 2>/dev/null || true

# Pre-download the HuggingFace embedding model into the image layer.
# This bakes the ~440 MB model weights so containers never download at runtime —
# works identically in local Docker and AWS ECS.
ENV HF_HOME=/root/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Disable HuggingFace network calls at runtime — model is already baked in above
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

# Copy project files
COPY . .


# Expose port
EXPOSE 8080

# Run FastAPI with uvicorn (production — no --reload, multi-worker)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]