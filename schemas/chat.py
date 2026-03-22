"""
Chat-related schemas for request/response models.
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    tenant_id: Optional[str] = "default"
    chatbot_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    tenant_id: str
    chatbot_id: Optional[str] = None
    session_id: Optional[str] = None
