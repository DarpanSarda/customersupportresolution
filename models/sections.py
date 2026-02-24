from pydantic import BaseModel, Field
from typing import Optional, List

class IntentModel(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class InputModel(BaseModel):
    raw_text: str

class DecisionSchema(BaseModel):
    action: str
    route: str
    reason: str

class UnderstandingSchema(BaseModel):
    """Schema for understanding section (Tier 1 output)."""
    intent: IntentModel
    input: InputModel

class LifecycleSchema(BaseModel):
    current_node: Optional[str]
    status: Optional[str]

class ToolCallModel(BaseModel):
    tool_name: str
    status: str
    input_payload: dict
    output_payload: Optional[dict]
    error: Optional[str]


class ExecutionSchema(BaseModel):
    tools_called: Optional[List[ToolCallModel]] = []
    final_response: Optional[str] = None
    response_confidence: Optional[float] = None