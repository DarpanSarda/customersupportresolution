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


# ============================================================
# RAG / Knowledge Retrieval Models (Tier 2)
# ============================================================

class RetrievedDocument(BaseModel):
    """Single retrieved document from vector store."""
    content: str = Field(..., description="Document text content")
    doc_id: str = Field(..., description="Unique document identifier")
    source: str = Field(..., description="Source type: faq, manual, sop, policy, etc.")
    metadata: dict = Field(default_factory=dict, description="Document metadata (tenant, category, etc.)")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (0-1)")
    stage: Optional[str] = Field(None, description="Which stage retrieved this: stage_1, stage_2, reranker")


class RAGSchema(BaseModel):
    """Schema for knowledge section (RAGAgent output).

    Contains retrieved documents from vector database with cascade retrieval:
    - Stage 1: bge-small (100 docs) - Fast, broad search
    - Stage 2: bge-base (50 docs) - Quality refinement
    - Stage 3: reranker (15 docs) - Final selection
    """
    query: str = Field(..., description="Original user query for retrieval")
    documents: List[RetrievedDocument] = Field(default_factory=list, description="Final retrieved documents")
    total_retrieved: int = Field(..., description="Number of documents retrieved")
    retrieval_method: str = Field(default="cascade", description="cascade, vector, hybrid")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Average relevance score")
    has_relevant_content: bool = Field(default=False, description="Whether relevant documents were found")

    # Cascade stage metadata
    stage_1_count: Optional[int] = Field(None, description="Stage 1 retrieval count")
    stage_2_count: Optional[int] = Field(None, description="Stage 2 retrieval count")
    reranker_count: Optional[int] = Field(None, description="After reranker count")
    retrieval_latency_ms: Optional[int] = Field(None, description="Total retrieval time")


# ============================================================
# Web Search Models (Tier 2)
# ============================================================

class WebSearchResult(BaseModel):
    """Single result from web search."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    content: str = Field(..., description="Snippet/content of the search result")
    score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance score")
    source: str = Field(default="web", description="Search provider: tavily, serper, etc.")
    published_date: Optional[str] = Field(None, description="Publication date if available")


class WebSearchSchema(BaseModel):
    """Schema for web_search section (WebSearchAgent output).

    Contains web search results when internal knowledge is insufficient.
    """
    query: str = Field(..., description="Original search query")
    results: List[WebSearchResult] = Field(default_factory=list, description="Web search results")
    total_results: int = Field(default=0, description="Number of results retrieved")
    search_provider: str = Field(default="tavily", description="Search provider used: tavily, serper")
    has_results: bool = Field(default=False, description="Whether results were found")
    search_latency_ms: Optional[int] = Field(None, description="Total search time")


# ============================================================
# Escalation Models (Tier 3)
# ============================================================

class EscalationSchema(BaseModel):
    """Schema for escalation section (EscalationAgent output).

    EscalationAgent builds the escalation payload AFTER DecisionAgent
    has already decided to escalate. It does NOT decide escalation,
    it only prepares the escalation metadata.

    Reads from:
    - decision.action: Must be "ESCALATE"
    - decision.route: Escalation reason route
    - decision.reason: Human-readable reason
    - understanding.sentiment: For priority calculation
    - policy.priority: Base priority level

    Writes to:
    - escalation section with structured escalation metadata
    """
    reason: str = Field(..., description="Why escalation was triggered (from decision.route)")
    priority: str = Field(..., description="Priority level: LOW, MEDIUM, HIGH, CRITICAL")
    status: str = Field(default="initiated", description="Escalation status: initiated, pending, completed")
    channel: str = Field(default="ticket_system", description="Escalation channel: ticket_system, email, slack, webhook")
    business_action: str = Field(default="CREATE_ESCALATION_TICKET", description="Business action for ToolExecutionAgent")

    # Context for human agent
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")
    summary: Optional[str] = Field(None, description="Brief summary for human agent")

    # Timestamps
    escalated_at: Optional[str] = Field(None, description="When escalation was initiated")


# ============================================================
# Vanna Text-to-SQL Models
# ============================================================

class VannaSchema(BaseModel):
    """Schema for vanna section (VannaAgent output).

    Text-to-SQL query generation and execution results.
    """
    status: str = Field(..., description="Query status: success, error")
    question: str = Field(..., description="Original natural language question")
    sql: Optional[str] = Field(None, description="Generated SQL query")
    results: Optional[list] = Field(default_factory=list, description="Query results as list of dicts")
    count: Optional[int] = Field(None, description="Number of rows returned")
    error: Optional[str] = Field(None, description="Error message if failed")