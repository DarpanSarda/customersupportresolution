"""
Response patch schemas for agent contributions.

Each agent contributes a patch/segment to the final response.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ResponsePatch(BaseModel):
    """
    A patch/segment contributed by an agent.

    Each agent adds their part to build the complete response.
    """

    agent_name: str = Field(..., description="Name of the agent contributing this patch")
    patch_type: str = Field(..., description="Type of patch (intent, sentiment, entity, response, etc.)")

    # Content
    content: Optional[str] = Field(None, description="Text content for the response")
    data: Optional[Dict[str, Any]] = Field(None, description="Structured data (e.g., entities, intent info)")

    # Metadata
    confidence: float = Field(1.0, description="Confidence score for this patch")
    timestamp: str = Field(..., description="When this patch was created")

    # Tool usage (if agent used a tool)
    tool_used: Optional[str] = Field(None, description="Tool that was used")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Result from tool execution")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class FinalResponse(BaseModel):
    """
    Final response built from all agent patches.

    Combines patches from all agents into the final output.
    """

    # Response text
    response: str = Field(..., description="Final response text to user")

    # Agent contributions
    patches: List[ResponsePatch] = Field(default_factory=list, description="All patches from agents")

    # Conversation data
    detected_intent: Optional[str] = Field(None, description="Detected intent")
    detected_sentiment: Optional[str] = Field(None, description="Detected sentiment")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")

    # Metadata
    tenant_id: str = Field(..., description="Tenant identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    chatbot_id: Optional[str] = Field(None, description="Chatbot identifier")

    # Tools used
    tools_used: List[str] = Field(default_factory=list, description="Tools that were used")

    # Escalation
    escalated: bool = Field(False, description="Whether conversation was escalated")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation")

    # Performance
    total_processing_time: float = Field(0.0, description="Total processing time in seconds")
    agent_timing: Dict[str, float] = Field(default_factory=dict, description="Time per agent")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
