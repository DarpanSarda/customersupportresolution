from pydantic import BaseModel , Field

class IntentAgentOutput(BaseModel):
    intent: str
    confidence: float = Field(..., ge=0.0, le=1.0)