"""
ConversationState - Shared state schema for agent orchestration.

This module defines the shared state object that gets passed between agents.
Each agent reads from and writes to specific fields in this state.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationState:
    """
    Shared conversation state passed between all agents.

    Each agent is responsible for only modifying its assigned fields:
    - IntentAgent: intent, intent_confidence
    - SentimentAgent: sentiment, sentiment_confidence
    - RAGRetrievalAgent: rag_results, rag_context, rag_confidence
    - PolicyAgent: policy_results, policy_action
    - ResponseAgent: response, response_type

    Agents must NEVER modify fields owned by other agents.
    """

    # ============ Input Fields ============
    # These fields are provided by the user/request and should NOT be modified by agents

    client_id: str
    """Client/Tenant identifier"""

    session_id: str
    """Session identifier for conversation tracking"""

    user_message: str
    """The current user message"""

    tenant_id: str
    """Tenant identifier for multi-tenancy"""

    # ============ Agent-Generated Fields ============
    # Each agent only modifies its own fields

    # IntentAgent output
    intent: Optional[str] = None
    """Detected intent label (e.g., FAQ_QUERY, REFUND_REQUEST)"""

    intent_confidence: float = 0.0
    """Confidence score for intent classification (0-1)"""

    intent_raw: Optional[Dict[str, Any]] = None
    """Raw intent agent output for debugging"""

    # SentimentAgent output
    sentiment: Optional[str] = None
    """Detected sentiment label (e.g., positive, neutral, negative)"""

    sentiment_confidence: float = 0.0
    """Confidence score for sentiment classification (0-1)"""

    sentiment_raw: Optional[Dict[str, Any]] = None
    """Raw sentiment agent output for debugging"""

    # RAGRetrievalAgent output
    rag_results: List[Dict[str, Any]] = field(default_factory=list)
    """List of retrieved passages from knowledge base"""

    rag_context: str = ""
    """Formatted context string for response generation"""

    rag_confidence: float = 0.0
    """Confidence score for RAG retrieval (0-1)"""

    rag_source_type: Optional[str] = None
    """Source type: 'faq', 'knowledge_base', or None"""

    rag_raw: Optional[Dict[str, Any]] = None
    """Raw RAG agent output for debugging"""

    # PolicyAgent output
    policy_results: Dict[str, Any] = field(default_factory=dict)
    """Policy evaluation results"""

    policy_action: Optional[str] = None
    """Recommended action based on policy"""

    policy_raw: Optional[Dict[str, Any]] = None
    """Raw policy agent output for debugging"""

    # ResponseAgent output
    response: Optional[str] = None
    """Final generated response to the user"""

    response_type: Optional[str] = None
    """Type of response: 'faq_answer', 'kb_answer', 'generic', etc."""

    # ============ Shared Context Fields ============
    # These fields can be read/written by multiple agents

    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    """Conversation history for context"""

    context_bundle: Dict[str, Any] = field(default_factory=dict)
    """Additional context collected during processing"""

    config: Dict[str, Any] = field(default_factory=dict)
    """Configuration values for agents"""

    # ============ Metadata ============

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    """Timestamp when state was created"""

    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    """Timestamp when state was last updated"""

    processing_time_ms: Optional[int] = None
    """Total processing time in milliseconds"""

    # ============ Error Handling ============

    errors: List[Dict[str, Any]] = field(default_factory=list)
    """List of errors encountered during processing"""

    warnings: List[Dict[str, Any]] = field(default_factory=list)
    """List of warnings during processing"""

    def add_error(self, agent_name: str, error: str, error_type: str = "processing_error"):
        """Add an error to the state."""
        self.errors.append({
            "agent": agent_name,
            "error": error,
            "type": error_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._update_timestamp()

    def add_warning(self, agent_name: str, warning: str):
        """Add a warning to the state."""
        self.warnings.append({
            "agent": agent_name,
            "warning": warning,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._update_timestamp()

    def _update_timestamp(self):
        """Update the timestamp."""
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "client_id": self.client_id,
            "session_id": self.session_id,
            "user_message": self.user_message,
            "tenant_id": self.tenant_id,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "sentiment": self.sentiment,
            "sentiment_confidence": self.sentiment_confidence,
            "rag_results": self.rag_results,
            "rag_context": self.rag_context,
            "rag_confidence": self.rag_confidence,
            "rag_source_type": self.rag_source_type,
            "policy_results": self.policy_results,
            "policy_action": self.policy_action,
            "response": self.response,
            "response_type": self.response_type,
            "conversation_history": self.conversation_history,
            "context_bundle": self.context_bundle,
            "errors": self.errors,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processing_time_ms": self.processing_time_ms
        }

    @classmethod
    def from_request(cls, request: Dict[str, Any]) -> "ConversationState":
        """
        Create ConversationState from a request dict.

        Args:
            request: Dict containing at least:
                - client_id/tenant_id
                - session_id
                - message/user_message

        Returns:
            ConversationState instance
        """
        return cls(
            client_id=request.get("client_id", request.get("tenant_id", "default")),
            session_id=request.get("session_id", ""),
            user_message=request.get("message", request.get("user_message", "")),
            tenant_id=request.get("tenant_id", request.get("client_id", "default"))
        )


class StateUpdate:
    """
    Wrapper for state updates returned by agents.

    Each agent should only return the fields it modifies.
    This wrapper ensures that only authorized fields are updated.

    Example:
        >>> state_update = StateUpdate(intent="FAQ_QUERY", intent_confidence=0.95)
        >>> state = state_update.apply(state)
    """

    def __init__(self, **updates):
        """
        Initialize with field updates.

        Args:
            **updates: Field names and their new values
        """
        self.updates = updates

    def apply(self, state: ConversationState) -> ConversationState:
        """
        Apply updates to state.

        Only updates fields that exist in the ConversationState dataclass.

        Args:
            state: Current conversation state

        Returns:
            Updated state (same instance, modified in place)
        """
        for field, value in self.updates.items():
            if hasattr(state, field):
                setattr(state, field, value)
            else:
                state.add_warning(
                    agent_name="StateUpdate",
                    warning=f"Attempted to update non-existent field: {field}"
                )

        state._update_timestamp()
        return state

    def to_dict(self) -> Dict[str, Any]:
        """Convert updates to dictionary."""
        return self.updates
