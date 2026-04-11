#!/bin/bash
set -e

# Start FastAPI backend on port 8080
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 2 &

# Start Streamlit frontend on port 8081 (points to local FastAPI)
API_URL=http://localhost:8080 streamlit run streamlit_app.py \
  --server.port 8081 \
  --server.address 0.0.0.0 \
  --server.headless true
