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
        build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (Python package/dependency manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_LINK_MODE=copy
ENV PYTHONPATH="/app:/app/multi_doc_chat"

# Copy dependency manifests for better layer caching
COPY requirements.txt ./

# Install dependencies
RUN uv pip install --system -r requirements.txt && \
    uv pip uninstall --system pinecone-plugin-inference pinecone-plugin-assistant 2>/dev/null || true

# Pre-bake FlashrankRerank model (~80MB) so the first request doesn't stall
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MultiBERT-L-12')"

# Copy project files
COPY . .


# Expose ports
EXPOSE 8080 8081

# Make startup script executable
RUN chmod +x start.sh

# Start FastAPI (8080) + Streamlit (8081) together
CMD ["bash", "start.sh"]