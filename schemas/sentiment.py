"""
Sentiment-related schemas for analysis results.
"""

from pydantic import BaseModel
from typing import Optional, List


class SentimentLabel(BaseModel):
    """Sentiment label configuration"""
    sentiment_label: str
    escalation_threshold: Optional[float] = None
    response_guideline: Optional[str] = None


class SentimentResult(BaseModel):
    """Result of sentiment analysis"""
    sentiment: str
    urgency_score: float
    toxicity_flag: bool
    confidence: float
    reasoning: Optional[str] = None
    emotional_indicators: Optional[List[str]] = None


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    message: str
    conversation_history: Optional[List[dict]] = []
    tenant_id: str = "default"
    session_id: Optional[str] = None
