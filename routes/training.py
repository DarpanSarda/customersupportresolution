"""
Training routes - Simple file ingestion (PDF, MD, TXT).
"""

from fastapi import APIRouter, UploadFile, Form
from services.TrainingRAGAgent import TrainingRAGAgent
import tempfile
import os

router = APIRouter(prefix="/training", tags=["training"])

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = TrainingRAGAgent()
    return _agent


@router.post("/")
async def train(file: UploadFile, tenant_id: str = Form("default")):
    """Train RAG with file (PDF, MD, TXT)."""
    agent = get_agent()

    # Save uploaded file to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await agent.ingest_from_file(tmp_path, tenant_id, file.filename)
        return result
    finally:
        os.unlink(tmp_path)
