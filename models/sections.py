from pydantic import BaseModel, Field
from typing import Optional

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
