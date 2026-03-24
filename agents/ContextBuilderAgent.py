"""
ContextBuilderAgent - Aggregates outputs from all previous agents.

This agent sits between PolicyAgent and ResponseAgent in the pipeline.
It collects all intermediate outputs and creates a structured context bundle
for the ResponseAgent to use when generating the final response.

According to the Agent Contract:
- MUST only read from shared state (intent, sentiment, rag_results, policy_results, etc.)
- MUST only return state updates (context_bundle field)
- MUST NOT call other agents
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from core.BaseAgent import BaseAgent
from core.ConversationState import ConversationState, StateUpdate


# Context Bundle Models
class UserQueryContext(BaseModel):
    """User's query context"""
    original_message: str = Field(..., description="Original user message")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    intent: str = Field(default="unknown", description="Classified intent")
    confidence: float = Field(default=0.0, description="Intent confidence score")


class UserContext(BaseModel):
    """User's emotional and profile context"""
    sentiment: str = Field(default="neutral", description="Detected sentiment")
    urgency: float = Field(default=0.5, description="Urgency score (0-1)")
    toxicity_flag: bool = Field(default=False, description="Toxic language detected")
    profile: Optional[Dict[str, Any]] = Field(None, description="User profile data")


class KnowledgeContext(BaseModel):
    """Retrieved knowledge context"""
    retrieved_passages: List[str] = Field(default_factory=list, description="RAG passages")
    source_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    relevance_scores: List[float] = Field(default_factory=list, description="Passage scores")
    total_passages: int = Field(default=0, description="Total passages retrieved")
    source_type: Optional[str] = Field(None, description="Source type: faq, knowledge_base, or None")


class PolicyContext(BaseModel):
    """Business policy context"""
    allowed_actions: List[str] = Field(default_factory=list, description="Permitted actions")
    denied_actions: List[str] = Field(default_factory=list, description="Prohibited actions")
    conditions: List[str] = Field(default_factory=list, description="Policy conditions")
    reasoning: Optional[str] = Field(None, description="Policy reasoning explanation")
    business_action: Optional[str] = Field(None, description="Recommended business action")
    priority: str = Field(default="normal", description="Priority level")


class ConversationContext(BaseModel):
    """Conversation history context"""
    history: List[Dict[str, str]] = Field(default_factory=list, description="Conversation turns")
    turn_count: int = Field(default=0, description="Number of conversation turns")
    summary: Optional[str] = Field(None, description="Conversation summary if long")


class ContextBundle(BaseModel):
    """Complete context bundle for response generation"""
    user_query: UserQueryContext
    user_context: UserContext
    knowledge: KnowledgeContext
    policy: PolicyContext
    conversation: ConversationContext
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    tenant_id: str = Field(default="default", description="Tenant identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="ISO timestamp of creation")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for state storage"""
        return {
            "user_query": self.user_query.dict(),
            "user_context": self.user_context.dict(),
            "knowledge": self.knowledge.dict(),
            "policy": self.policy.dict(),
            "conversation": self.conversation.dict(),
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp
        }


class ContextBuilderAgent(BaseAgent):
    """
    Aggregates outputs from all previous agents into a structured context bundle.

    This agent is responsible for:
    1. Collecting outputs from IntentAgent, SentimentAgent, RAGAgent, PolicyAgent
    2. Structuring the data into a clean, hierarchical format
    3. Optionally optimizing the context (filtering, summarization)
    4. Providing the context bundle to ResponseAgent

    The agent does NOT call any tools - it's a pure aggregator/transformer.

    State Dependencies:
        - user_message: Original user message
        - intent: Detected intent classification
        - intent_confidence: Intent confidence score
        - sentiment: Detected sentiment
        - sentiment_confidence: Sentiment confidence/urgency score
        - rag_results: Retrieved passages from RAG
        - rag_context: Formatted RAG context
        - rag_confidence: RAG retrieval confidence
        - rag_source_type: RAG source type
        - policy_results: Policy evaluation results
        - policy_action: Recommended business action
        - conversation_history: Conversation history
        - context_bundle: Existing context (may have entities from IntentAgent)

    State Updates:
        - context_bundle: Structured context object
    """

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry = None,
        enable_optimization: bool = False
    ):
        """
        Initialize ContextBuilderAgent.

        Args:
            llm_client: LLM client for optional optimization
            system_prompt: Custom system prompt (optional)
            config: Additional configuration
            tool_registry: Tool registry (not used by this agent)
            enable_optimization: Whether to use LLM for context optimization
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry
        )
        self.enable_optimization = enable_optimization
        print(f"[DEBUG] ContextBuilderAgent initialized with enable_optimization={enable_optimization}")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a Context Builder Agent. Your role is to structure and optimize context for response generation.

## Your Responsibilities

1. **Collect**: Gather all outputs from previous agents (Intent, Sentiment, RAG, Policy)
2. **Structure**: Organize the data into a clean, hierarchical format
3. **Optimize**: Filter low-relevance information and summarize long conversations
4. **Prioritize**: Ensure the most important context is highlighted for response generation

## Context Structure

Your output should have these sections:
- **user_query**: Original message, entities, intent, confidence
- **user_context**: Sentiment, urgency, toxicity, profile
- **knowledge**: Retrieved passages with sources and scores
- **policy**: Allowed/denied actions, conditions, reasoning
- **conversation**: History, turn count, optional summary

## Optimization Rules

- Keep all user intent and sentiment data (critical for response tone)
- Keep passages with relevance score >= 0.5
- For conversations longer than 10 turns, summarize older turns
- Preserve all policy constraints and allowed actions
- Maintain all extracted entities

## Output Format

Provide a brief optimization summary if changes are needed, otherwise respond with "CONTEXT_OPTIMAL"."""

    async def process(self, input_data: Dict[str, Any], **kwargs) -> StateUpdate:
        """
        Build context bundle from all previous agent outputs.

        Args:
            input_data: ConversationState instance or dict with state fields
            **kwargs: Additional parameters

        Returns:
            StateUpdate with context_bundle field populated
        """
        print(f"[DEBUG] ContextBuilderAgent.process called")

        # Handle both ConversationState and dict input
        if isinstance(input_data, ConversationState):
            state = input_data
        elif isinstance(input_data, dict) and "state" in input_data and isinstance(input_data["state"], ConversationState):
            # Orchestrator passes state as a dict key - use it directly
            state = input_data["state"]
            print(f"[DEBUG] ContextBuilderAgent: using state from input_data dict")
        else:
            # Legacy dict format - convert to state-like object
            state = self._dict_to_state(input_data)

        # Phase 1: Collection
        user_query_context = self._build_user_query_context(state)
        user_context = self._build_user_context(state)
        knowledge_context = self._build_knowledge_context(state)
        policy_context = self._build_policy_context(state)
        conversation_context = self._build_conversation_context(state)

        # Phase 2: Create bundle
        context_bundle = ContextBundle(
            user_query=user_query_context,
            user_context=user_context,
            knowledge=knowledge_context,
            policy=policy_context,
            conversation=conversation_context,
            metadata={
                "session_id": state.session_id,
                "client_id": state.client_id,
                "agent_execution_order": self.config.get("agent_execution_order", [])
            },
            tenant_id=state.tenant_id,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

        print(f"[DEBUG] ContextBundle created with {context_bundle.knowledge.total_passages} passages")

        # Phase 3: Optional optimization
        if self.enable_optimization:
            context_bundle = await self._optimize_context(context_bundle, state)
            print(f"[DEBUG] ContextBundle optimization applied")

        # Create state update
        return StateUpdate(
            context_bundle=context_bundle.to_dict()
        )

    def _build_user_query_context(self, state: ConversationState) -> UserQueryContext:
        """Build user query context from state"""
        # Get entities from existing context_bundle (may have been populated by IntentAgent)
        existing_context = state.context_bundle or {}
        entities = existing_context.get("entities", {})

        return UserQueryContext(
            original_message=state.user_message,
            entities=entities,
            intent=state.intent or "unknown",
            confidence=state.intent_confidence
        )

    def _build_user_context(self, state: ConversationState) -> UserContext:
        """Build user context from state"""
        sentiment_raw = state.sentiment_raw or {}

        return UserContext(
            sentiment=state.sentiment or "neutral",
            urgency=state.sentiment_confidence,
            toxicity_flag=sentiment_raw.get("toxicity_flag", False),
            profile=None  # User profile not in current state schema
        )

    def _build_knowledge_context(self, state: ConversationState) -> KnowledgeContext:
        """Build knowledge context from RAG results"""
        # Handle multiple formats for rag_results:
        # 1. Dict with 'retrieved_passages', 'source_ids', 'relevance_scores'
        # 2. List of dicts with 'passage'/'text' keys
        # 3. List of strings (simple passages only)
        rag_results = state.rag_results
        passages = []
        source_ids = []
        scores = []

        if isinstance(rag_results, dict):
            # New format: dict with 'retrieved_passages', 'source_ids', 'relevance_scores'
            passages = rag_results.get("retrieved_passages", [])
            source_ids = rag_results.get("source_ids", [])
            scores = rag_results.get("relevance_scores", [])
        elif isinstance(rag_results, list):
            # Check if list contains dicts or strings
            if rag_results and isinstance(rag_results[0], dict):
                # List of dicts (legacy format)
                for item in rag_results:
                    content = item.get("passage") or item.get("content") or item.get("text", "")
                    passages.append(content)
                    source_ids.append(item.get("source_id", item.get("source", "")))
                    scores.append(item.get("score", item.get("relevance_score", 0.0)))
            elif rag_results and isinstance(rag_results[0], str):
                # List of strings (simple passages from RAGRetrievalAgent)
                passages = rag_results
                source_ids = [""] * len(rag_results)  # Empty source IDs
                scores = [0.0] * len(rag_results)  # Zero scores

        # Also check rag_raw for detailed results (if available)
        if state.rag_raw and isinstance(state.rag_raw, dict):
            # Extract detailed results if available
            detailed_results = state.rag_raw.get("results", [])
            if detailed_results and isinstance(detailed_results, list):
                # Update source_ids and scores from detailed results
                for i, result in enumerate(detailed_results):
                    if i < len(source_ids) and isinstance(result, dict):
                        if result.get("id"):
                            source_ids[i] = result.get("id", source_ids[i])
                        if "score" in result or "rerank_score" in result:
                            scores[i] = result.get("rerank_score") or result.get("score", scores[i])

        print(f"[DEBUG] ContextBundle created with {len(passages)} passages")
        return KnowledgeContext(
            retrieved_passages=passages,
            source_ids=source_ids,
            relevance_scores=scores,
            total_passages=len(passages),
            source_type=state.rag_source_type
        )

    def _build_policy_context(self, state: ConversationState) -> PolicyContext:
        """Build policy context from policy results"""
        policy_data = state.policy_results or {}

        return PolicyContext(
            allowed_actions=policy_data.get("allowed_actions", []),
            denied_actions=policy_data.get("denied_actions", []),
            conditions=policy_data.get("conditions", []),
            reasoning=policy_data.get("reasoning"),
            business_action=state.policy_action,
            priority=policy_data.get("priority", "normal")
        )

    def _build_conversation_context(self, state: ConversationState) -> ConversationContext:
        """Build conversation context from history"""
        history = state.conversation_history or []

        # Optionally summarize long conversations
        summary = None
        if len(history) > 10:
            summary = self._create_summary(history)

        return ConversationContext(
            history=history,
            turn_count=len(history),
            summary=summary
        )

    def _create_summary(self, history: List[Dict[str, str]]) -> str:
        """Create a simple summary of conversation history"""
        if not history:
            return ""

        # Extract key topics from recent messages
        recent = history[-5:] if len(history) >= 5 else history
        topics = []

        for turn in recent:
            user_msg = turn.get("user", turn.get("message", ""))
            if isinstance(user_msg, str):
                user_msg_lower = user_msg.lower()
                if "refund" in user_msg_lower:
                    topics.append("refund discussion")
                elif "order" in user_msg_lower:
                    topics.append("order inquiry")
                elif "complaint" in user_msg_lower or "issue" in user_msg_lower:
                    topics.append("complaint raised")

        if topics:
            return f"Conversation covers: {', '.join(set(topics))}"
        return "General customer inquiry"

    async def _optimize_context(
        self,
        context_bundle: ContextBundle,
        state: ConversationState
    ) -> ContextBundle:
        """
        Optionally optimize context using LLM.

        This can:
        - Filter low-relevance passages
        - Summarize long history
        - Prioritize information
        """
        # For now, return as-is (optimization can be enhanced later)
        return context_bundle

    def _dict_to_state(self, data: Dict[str, Any]) -> Any:
        """
        Convert dict to state-like object for backward compatibility.

        Args:
            data: Dictionary with state fields

        Returns:
            Simple object with state attributes
        """
        class SimpleState:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return SimpleState(
            client_id=data.get("client_id", "default"),
            session_id=data.get("session_id", ""),
            user_message=data.get("user_message", ""),
            tenant_id=data.get("tenant_id", "default"),
            intent=data.get("intent"),
            intent_confidence=data.get("intent_confidence", 0.0),
            sentiment=data.get("sentiment"),
            sentiment_confidence=data.get("sentiment_confidence", 0.0),
            sentiment_raw=data.get("sentiment_raw"),
            rag_results=data.get("rag_results", []),
            rag_context=data.get("rag_context", ""),
            rag_confidence=data.get("rag_confidence", 0.0),
            rag_source_type=data.get("rag_source_type"),
            rag_raw=data.get("rag_raw"),  # Added rag_raw
            policy_results=data.get("policy_results", {}),
            policy_action=data.get("policy_action"),
            policy_raw=data.get("policy_raw"),  # Added policy_raw for consistency
            conversation_history=data.get("conversation_history", []),
            context_bundle=data.get("context_bundle", {}),
            config=data.get("config", {}),
            add_error=lambda *args: None
        )

    @classmethod
    def get_agent_info(cls) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary with agent details
        """
        return {
            "name": "ContextBuilderAgent",
            "description": "Aggregates outputs from all previous agents into a structured context bundle",
            "input_fields": [
                "user_message", "intent", "intent_confidence",
                "sentiment", "sentiment_confidence", "sentiment_raw",
                "rag_results", "rag_context", "rag_confidence", "rag_source_type",
                "policy_results", "policy_action",
                "conversation_history", "context_bundle"
            ],
            "output_fields": ["context_bundle"],
            "state_modifications": ["context_bundle"]
        }
