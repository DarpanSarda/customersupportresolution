from pydantic import BaseModel, Field
from typing import Optional, List

class IntentModel(BaseModel):
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SentimentModel(BaseModel):
    """Sentiment analysis result with customer emotional context."""
    label: str = Field(..., description="Sentiment label (e.g., POSITIVE, NEGATIVE, NEUTRAL, FRUSTRATED, ANGRY)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    intensity: Optional[float] = Field(None, ge=0.0, le=1.0, description="Emotional intensity 0-1")
    indicators: Optional[List[str]] = Field(default_factory=list, description="Key phrases/words indicating sentiment")


class InputModel(BaseModel):
    raw_text: str

class DecisionSchema(BaseModel):
    action: str
    route: str
    reason: str

class UnderstandingSchema(BaseModel):
    """Schema for understanding section (Tier 1 output)."""
    intent: IntentModel
    sentiment: Optional[SentimentModel] = None
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


class PolicyViolation(BaseModel):
    """Single policy violation record."""
    policy_name: str = Field(..., description="Name of the violated policy")
    severity: str = Field(..., description="Severity level: LOW, MEDIUM, HIGH, CRITICAL")
    reason: str = Field(..., description="Explanation of the violation")


class PolicyModel(BaseModel):
    """Policy evaluation result."""
    compliant: bool = Field(..., description="Whether the request complies with all applicable policies")
    applicable_policies: List[str] = Field(default_factory=list, description="List of policies that were evaluated")
    violations: List[PolicyViolation] = Field(default_factory=list, description="List of any policy violations")
    restrictions: List[str] = Field(default_factory=list, description="Any restrictions applied based on policies")
    reason: str = Field(..., description="Overall explanation of policy evaluation result")


class PolicyViolationSchema(BaseModel):
    """Schema for policy violation in state (dict format)."""
    policy_name: str
    severity: str
    reason: str


class PolicySchema(BaseModel):
    """Schema for policy section (Tier 2 output)."""
    compliant: bool
    applicable_policies: List[str] = Field(default_factory=list)
    violations: List[PolicyViolationSchema] = Field(default_factory=list)
    restrictions: List[str] = Field(default_factory=list)
    reason: str