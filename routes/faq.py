"""
FAQ routes - FAQ file ingestion (PDF, MD, TXT).
"""

from fastapi import APIRouter, UploadFile, Form
from services.FAQService import FAQService
import tempfile
import os

router = APIRouter(prefix="/faq", tags=["faq"])

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = FAQService()
    return _agent


@router.post("/")
async def ingest_faq(file: UploadFile, tenant_id: str = Form("default")):
    """Ingest FAQ file (PDF, MD, TXT) into faq_{tenant_id} collection."""
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


@router.get("/info")
async def get_faq_info(tenant_id: str = "default"):
    """Get FAQ collection information."""
    agent = get_agent()
    return agent.get_collection_info(tenant_id)
