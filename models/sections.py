from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

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
    """Record of a tool execution with async support."""
    tool_name: str
    status: str = Field(..., description="success | failed | pending")
    input_payload: dict
    output_payload: Optional[dict] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    execution_time_ms: Optional[int] = Field(None, description="Execution duration")
    async_job_id: Optional[str] = Field(None, description="Job ID if async operation")
    tenant_id: Optional[str] = Field(None, description="Tenant who initiated call")


class ExecutionSchema(BaseModel):
    tools_called: Optional[List[ToolCallModel]] = []
    final_response: Optional[str] = None
    response_confidence: Optional[float] = None
    async_jobs_pending: Optional[List[str]] = Field(default_factory=list, description="IDs of pending async jobs")


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
    """Schema for policy section (Tier 2 output) - Generic policy evaluation.

    PolicyAgent evaluates intent against configured policies and outputs
    abstract business actions. DecisionAgent maps business_action to concrete actions.
    """
    business_action: Optional[str] = Field(None, description="Abstract business action to perform")
    required_fields: List[str] = Field(default_factory=list, description="Fields required for this action")
    missing_fields: List[str] = Field(default_factory=list, description="Fields missing from context")
    allow_execution: bool = Field(False, description="Whether execution should proceed")
    escalate: bool = Field(False, description="Whether to escalate to human")
    priority: str = Field("normal", description="Priority level: low, normal, high, urgent")
    reason: str = Field(..., description="Explanation of policy decision")


class ContextSchema(BaseModel):
    """Schema for context section (ContextBuilderAgent output).

    Contains extracted entities from user messages based on tenant-specific schemas.
    """
    tenant_id: Optional[str] = Field(None, description="Tenant identifier for schema lookup")
    entities: dict = Field(default_factory=dict, description="Extracted entities as key-value pairs")
    extraction_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in extraction completeness")
    extracted_fields: List[str] = Field(default_factory=list, description="List of successfully extracted field names")
    extraction_reason: Optional[str] = Field(None, description="Explanation of extraction result")