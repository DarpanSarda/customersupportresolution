"""
Conversation state schemas for LangGraph.

Defines the state object that flows through the agent graph.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    """
    Complete conversation state that flows through the agent graph.

    This state is passed between agents and gets updated at each step.
    """

    # Input
    message: str = Field(..., description="User's input message")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    chatbot_id: Optional[str] = Field(None, description="Chatbot identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

    # Intent
    detected_intent: Optional[str] = Field(None, description="Detected intent from user message")
    intent_confidence: float = Field(0.0, description="Confidence score for intent")
    intent_meets_threshold: bool = Field(False, description="Whether intent meets confidence threshold")

    # Sentiment
    detected_sentiment: Optional[str] = Field(None, description="Detected sentiment (POSITIVE, NEUTRAL, NEGATIVE, etc.)")
    sentiment_score: float = Field(0.0, description="Sentiment score")

    # Entities
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities from conversation")

    # Policy
    business_action: Optional[str] = Field(None, description="Business action to execute")
    policy_result: Optional[Dict[str, Any]] = Field(None, description="Result of policy execution")

    # Response
    response: Optional[str] = Field(None, description="Final response to user")
    tool_used: Optional[str] = Field(None, description="Tool that was used")

    # Escalation
    should_escalate: bool = Field(False, description="Whether conversation should be escalated")
    escalation_reason: Optional[str] = Field(None, description="Reason for escalation")
    escalation_priority: Optional[str] = Field(None, description="Priority level (LOW, MEDIUM, HIGH, CRITICAL)")

    # Conversation flow
    current_node: str = Field(default="entry", description="Current node in the graph")
    next_node: Optional[str] = Field(None, description="Next node to visit")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Error handling
    error: Optional[str] = Field(None, description="Error message if something went wrong")
    retry_count: int = Field(0, description="Number of retries attempted")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class GraphConfig(BaseModel):
    """
    Configuration for the LangGraph system.

    Defines graph structure, nodes, edges, and routing logic.
    """

    # Graph definition
    graph_name: str = Field("customer_support_graph", description="Name of the graph")
    entry_node: str = Field("intent_agent", description="Starting node")

    # Nodes
    nodes: List[str] = Field(
        default=[
            "intent_agent",
            "sentiment_agent",
            "entity_agent",
            "policy_agent",
            "tool_agent",
            "response_agent",
            "escalation_agent",
            "fallback_agent"
        ],
        description="List of all nodes in the graph"
    )

    # Terminal nodes (end points)
    terminal_nodes: List[str] = Field(
        default=["response_agent", "escalation_agent", "fallback_agent"],
        description="Nodes that end the conversation flow"
    )

    # Routing rules (edges with conditions)
    routing_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Routing rules for conditional edges"
    )

    # Agent configurations (will be loaded from database)
    agent_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Configuration for each agent"
    )

    # Timeout configurations
    node_timeout: int = Field(30, description="Timeout per node in seconds")
    max_retries: int = Field(3, description="Maximum retry attempts per node")

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
