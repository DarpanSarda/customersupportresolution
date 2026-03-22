"""
Intent-related schemas for classification results.
"""

from pydantic import BaseModel
from typing import Optional, List


class IntentLabel(BaseModel):
    """Intent label configuration"""
    label: str
    description: Optional[str] = None
    confidence_threshold: float = 0.7
    tool_mapping: Optional[str] = None
    examples: Optional[List[str]] = None


class IntentResult(BaseModel):
    """Result of intent classification"""
    intent: str
    confidence: float
    meets_threshold: bool
    tool_mapping: Optional[str] = None
    raw_response: Optional[dict] = None


class IntentClassificationRequest(BaseModel):
    """Request for intent classification"""
    message: str
    tenant_id: str = "default"
    session_id: Optional[str] = None
    available_intents: Optional[List[IntentLabel]] = None
