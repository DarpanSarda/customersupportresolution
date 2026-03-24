# ContextBuilderAgent Implementation Plan

## Overview

The **ContextBuilderAgent** is a Tier 2 Knowledge & Decision Agent responsible for assembling intermediate outputs from all previous agents into a structured context bundle. This agent serves as the critical bridge between data collection/analysis and response generation.

### Why ContextBuilderAgent is Needed

1. **Prevents Prompt Overload**: Instead of passing raw outputs from multiple agents directly to the ResolutionAgent, ContextBuilderAgent creates a clean, structured bundle
2. **Maintains Separation of Concerns**: ResolutionAgent focuses on response generation, not data aggregation
3. **Enables Context Optimization**: Can prioritize, filter, or transform data before response generation
4. **Improves Debugging**: Single point to inspect what context is being passed to response generation
5. **Supports Multi-Tenancy**: Can apply tenant-specific context transformations

### Architecture Position

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Agent Execution Flow                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐        │
│  │   Intent     │───▶│   Sentiment   │───▶│     RAG      │        │
│  │    Agent     │    │    Agent      │    │    Agent     │        │
│  └──────────────┘    └───────────────┘    └──────────────┘        │
│                                                      │              │
│                                                  ┌───▼──────────┐  │
│  ┌──────────────┐                            │  Policy      │  │
│  │  User Input  │                            │  Agent       │  │
│  └──────┬───────┘                            └──────┬───────┘  │
│         │                                           │           │
│         │         ┌─────────────────────────────────▼───────┐   │
│         └────────▶│         ContextBuilderAgent            │   │
│                   │   (Aggregates all outputs)             │   │
│                   └─────────────────────────────────┬───────┘   │
│                                                     │           │
│                   ┌─────────────────────────────────▼───────┐   │
│                   │          ResolutionAgent                │   │
│                   │    (Uses structured context bundle)     │   │
│                   └─────────────────────────────────┬───────┘   │
│                                                     │           │
│                   ┌─────────────────────────────────▼───────┐   │
│                   │         EscalationAgent                │   │
│                   └─────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Table of Contents

1. [Agent Specification](#agent-specification)
2. [Data Flow](#data-flow)
3. [Context Bundle Structure](#context-bundle-structure)
4. [Implementation Phases](#implementation-phases)
5. [Code Implementation](#code-implementation)
6. [Integration Steps](#integration-steps)
7. [Testing Strategy](#testing-strategy)
8. [Verification Checklist](#verification-checklist)

---

## Agent Specification

### Basic Information

| Property | Value |
|----------|-------|
| **Agent Name** | ContextBuilderAgent |
| **Tier** | Tier 2 - Knowledge & Decision |
| **Position** | After PolicyAgent, Before ResolutionAgent |
| **Tools Required** | None (aggregator only) |
| **Async** | Yes |
| **State Modification** | Writes to `context_bundle` field |

### Input Sources

The ContextBuilderAgent reads from the following state fields:

| Field | Source Agent | Description |
|-------|--------------|-------------|
| `intent` | IntentAgent | Classified intent with confidence |
| `sentiment` | SentimentAgent | Sentiment label and urgency score |
| `rag_results` | RAGRetrievalAgent | Retrieved passages and relevance scores |
| `policy_results` | PolicyAgent | Allowed/denied actions and conditions |
| `user_message` | User Input | Original user message |
| `conversation_history` | Session | Previous conversation turns |
| `user_profile` | Session | User metadata (if available) |
| `entities` | IntentAgent/Extraction | Extracted entities from message |

### Output Structure

The ContextBuilderAgent writes to:

| Field | Type | Description |
|-------|------|-------------|
| `context_bundle` | ContextBundle | Structured context object (see below) |

---

## Data Flow

### Phase 1: Collection

```
┌─────────────────────────────────────────────────────────────┐
│  1. Collection Phase                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  intent = state.intent                                      │
│  sentiment = state.sentiment                                │
│  rag_results = state.rag_results                            │
│  policy_results = state.policy_results                      │
│  user_message = state.user_message                          │
│  conversation_history = state.conversation_history          │
│  user_profile = state.user_profile                          │
│  entities = state.entities                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Structuring

```
┌─────────────────────────────────────────────────────────────┐
│  2. Structuring Phase                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  context_bundle = {                                         │
│    "user_query": {                                          │
│      "original_message": user_message,                      │
│      "entities": entities,                                  │
│      "intent": intent.intent,                               │
│      "confidence": intent.confidence_score                  │
│    },                                                       │
│    "user_context": {                                        │
│      "sentiment": sentiment.sentiment,                      │
│      "urgency": sentiment.urgency_score,                    │
│      "profile": user_profile                                │
│    },                                                       │
│    "knowledge": {                                           │
│      "retrieved_passages": rag_results.passages,            │
│      "source_ids": rag_results.source_ids,                  │
│      "relevance_scores": rag_results.scores                 │
│    },                                                       │
│    "policy": {                                              │
│      "allowed_actions": policy_results.allowed_actions,     │
│      "denied_actions": policy_results.denied_actions,       │
│      "conditions": policy_results.conditions                │
│    },                                                       │
│    "conversation": {                                        │
│      "history": conversation_history,                       │
│      "turn_count": len(conversation_history)                │
│    }                                                        │
│  }                                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Optimization

```
┌─────────────────────────────────────────────────────────────┐
│  3. Optimization Phase (Optional)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  • Filter low-relevance passages (< 0.5 score)              │
│  • Prioritize recent conversation turns                     │
│  • Summarize long history (> 10 turns)                      │
│  • Apply tenant-specific transformations                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Context Bundle Structure

### Complete Schema

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class UserQueryContext(BaseModel):
    """User's query context"""
    original_message: str = Field(..., description="Original user message")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Intent confidence score")

class UserContext(BaseModel):
    """User's emotional and profile context"""
    sentiment: str = Field(..., description="Detected sentiment")
    urgency: float = Field(default=0.5, description="Urgency score (0-1)")
    toxicity_flag: bool = Field(default=False, description="Toxic language detected")
    profile: Optional[Dict[str, Any]] = Field(None, description="User profile data")

class KnowledgeContext(BaseModel):
    """Retrieved knowledge context"""
    retrieved_passages: List[str] = Field(default_factory=list, description="RAG passages")
    source_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    relevance_scores: List[float] = Field(default_factory=list, description="Passage scores")
    total_passages: int = Field(default=0, description="Total passages retrieved")

class PolicyContext(BaseModel):
    """Business policy context"""
    allowed_actions: List[str] = Field(default_factory=list, description="Permitted actions")
    denied_actions: List[str] = Field(default_factory=list, description="Prohibited actions")
    conditions: List[str] = Field(default_factory=list, description="Policy conditions")
    reasoning: Optional[str] = Field(None, description="Policy reasoning explanation")

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
    timestamp: str = Field(..., description="ISO timestamp of creation")
```

---

## Implementation Phases

### Phase 1: Agent Implementation

**Objective**: Create the ContextBuilderAgent class with proper state management.

**Files to Create**:
- `agents/ContextBuilderAgent.py`

**Dependencies**:
- `core/BaseAgent.py` - Base agent class
- `models/state.py` - State management models
- `core/PromptLoader.py` - Prompt template loader

### Phase 2: Prompt Template

**Objective**: Create prompt template for any LLM-based context optimization.

**Files to Create**:
- `prompts/ContextBuilderAgent/v1.txt`

**Purpose**: While ContextBuilderAgent is primarily a structural aggregator, the prompt enables optional LLM-based context optimization (summarization, filtering, prioritization).

### Phase 3: Orchestrator Integration

**Objective**: Integrate ContextBuilderAgent into the orchestrator execution flow.

**Files to Modify**:
- `core/Orchestrator.py` - Add ContextBuilderAgent to execution graph
- `core/AgentFactory.py` - Add factory method for ContextBuilderAgent
- `services/ChatService.py` - Initialize ContextBuilderAgent in chat pipeline

### Phase 4: Testing

**Objective**: Create comprehensive unit and integration tests.

**Files to Create**:
- `tests/test_context_builder_agent.py`

---

## Code Implementation

### Agent Implementation

**File**: `agents/ContextBuilderAgent.py`

```python
"""
ContextBuilderAgent - Aggregates outputs from all previous agents.

This agent sits between PolicyAgent and ResolutionAgent in the pipeline.
It collects all intermediate outputs and creates a structured context bundle
for the ResolutionAgent to use when generating the final response.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from core.BaseAgent import BaseAgent
from models.state import ConversationState, StateUpdate
from core.PromptLoader import PromptLoader
from llms.BaseLLM import BaseLLM


# Context Bundle Models
class UserQueryContext(BaseModel):
    """User's query context"""
    original_message: str = Field(..., description="Original user message")
    entities: Dict[str, Any] = Field(default_factory=dict, description="Extracted entities")
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., description="Intent confidence score")


class UserContext(BaseModel):
    """User's emotional and profile context"""
    sentiment: str = Field(..., description="Detected sentiment")
    urgency: float = Field(default=0.5, description="Urgency score (0-1)")
    toxicity_flag: bool = Field(default=False, description="Toxic language detected")
    profile: Optional[Dict[str, Any]] = Field(None, description="User profile data")


class KnowledgeContext(BaseModel):
    """Retrieved knowledge context"""
    retrieved_passages: List[str] = Field(default_factory=list, description="RAG passages")
    source_ids: List[str] = Field(default_factory=list, description="Source document IDs")
    relevance_scores: List[float] = Field(default_factory=list, description="Passage scores")
    total_passages: int = Field(default=0, description="Total passages retrieved")


class PolicyContext(BaseModel):
    """Business policy context"""
    allowed_actions: List[str] = Field(default_factory=list, description="Permitted actions")
    denied_actions: List[str] = Field(default_factory=list, description="Prohibited actions")
    conditions: List[str] = Field(default_factory=list, description="Policy conditions")
    reasoning: Optional[str] = Field(None, description="Policy reasoning explanation")


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
    timestamp: str = Field(..., description="ISO timestamp of creation")


class ContextBuilderAgent(BaseAgent):
    """
    Aggregates outputs from all previous agents into a structured context bundle.

    This agent is responsible for:
    1. Collecting outputs from IntentAgent, SentimentAgent, RAGAgent, PolicyAgent
    2. Structuring the data into a clean, hierarchical format
    3. Optionally optimizing the context (filtering, summarization)
    4. Providing the context bundle to ResolutionAgent

    The agent does NOT call any tools - it's a pure aggregator/transformer.
    """

    def __init__(
        self,
        llm_client: BaseLLM,
        prompt_loader: PromptLoader,
        enable_optimization: bool = False
    ):
        """
        Initialize ContextBuilderAgent.

        Args:
            llm_client: LLM client for optional optimization
            prompt_loader: Prompt template loader
            enable_optimization: Whether to use LLM for context optimization
        """
        super().__init__(
            name="ContextBuilderAgent",
            llm_client=llm_client,
            prompt_loader=prompt_loader
        )
        self.enable_optimization = enable_optimization

    async def process(self, state: ConversationState) -> StateUpdate:
        """
        Build context bundle from all previous agent outputs.

        Args:
            state: Current conversation state

        Returns:
            StateUpdate with context_bundle field populated
        """
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
                "chatbot_id": state.chatbot_id,
                "agent_execution_order": state.agent_execution_order
            },
            tenant_id=state.tenant_id,
            timestamp=datetime.utcnow().isoformat()
        )

        # Phase 3: Optional optimization
        if self.enable_optimization:
            context_bundle = await self._optimize_context(context_bundle, state)

        # Create state update
        return StateUpdate(
            agent_name=self.name,
            context_bundle=context_bundle.dict()
        )

    def _build_user_query_context(self, state: ConversationState) -> UserQueryContext:
        """Build user query context from state"""
        intent_data = state.intent or {}

        return UserQueryContext(
            original_message=state.user_message,
            entities=state.entities or {},
            intent=intent_data.get("intent", "unknown"),
            confidence=intent_data.get("confidence_score", 0.0)
        )

    def _build_user_context(self, state: ConversationState) -> UserContext:
        """Build user context from state"""
        sentiment_data = state.sentiment or {}

        return UserContext(
            sentiment=sentiment_data.get("sentiment", "neutral"),
            urgency=sentiment_data.get("urgency_score", 0.5),
            toxicity_flag=sentiment_data.get("toxicity_flag", False),
            profile=state.user_profile
        )

    def _build_knowledge_context(self, state: ConversationState) -> KnowledgeContext:
        """Build knowledge context from RAG results"""
        rag_data = state.rag_results or {}

        passages = rag_data.get("retrieved_passages", [])
        source_ids = rag_data.get("source_ids", [])
        scores = rag_data.get("relevance_scores", [])

        return KnowledgeContext(
            retrieved_passages=passages,
            source_ids=source_ids,
            relevance_scores=scores,
            total_passages=len(passages)
        )

    def _build_policy_context(self, state: ConversationState) -> PolicyContext:
        """Build policy context from policy results"""
        policy_data = state.policy_results or {}

        return PolicyContext(
            allowed_actions=policy_data.get("allowed_actions", []),
            denied_actions=policy_data.get("denied_actions", []),
            conditions=policy_data.get("conditions", []),
            reasoning=policy_data.get("policy_reasoning")
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
            user_msg = turn.get("user", "")
            if "refund" in user_msg.lower():
                topics.append("refund discussion")
            elif "order" in user_msg.lower():
                topics.append("order inquiry")
            elif "complaint" in user_msg.lower():
                topics.append("complaint raised")

        return f"Conversation covers: {', '.join(topics) if topics else 'general inquiry'}"

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
        # Load optimization prompt
        prompt = self.prompt_loader.load_prompt(
            agent_name="context_builder",
            version="v1"
        )

        # For now, return as-is (optimization can be enhanced later)
        return context_bundle
```

### AgentFactory Integration

**File**: `core/AgentFactory.py` (MODIFY)

Add the factory method:

```python
def create_context_builder_agent(
    self,
    enable_optimization: bool = False
) -> ContextBuilderAgent:
    """
    Create ContextBuilderAgent instance.

    Args:
        enable_optimization: Whether to enable LLM-based optimization

    Returns:
        ContextBuilderAgent instance
    """
    from agents.ContextBuilderAgent import ContextBuilderAgent

    return ContextBuilderAgent(
        llm_client=self.llm_client,
        prompt_loader=self.prompt_loader,
        enable_optimization=enable_optimization
    )
```

### Orchestrator Integration

**File**: `core/Orchestrator.py` (MODIFY)

Update the execution graph:

```python
# In __init__ method, update execution order
self.execution_order = [
    "intent",
    "sentiment",
    "rag",
    "policy",
    "context_builder",  # NEW
    "response",
    "escalation"
]

# In _build_execution_graph method, add edges
self.execution_graph = {
    "intent": ["sentiment"],
    "sentiment": ["rag"],
    "rag": ["policy"],
    "policy": ["context_builder"],  # MODIFIED
    "context_builder": ["response"],  # NEW
    "response": ["escalation"],
    "escalation": []  # Terminal node
}
```

### ChatService Integration

**File**: `services/ChatService.py` (MODIFY)

Add ContextBuilderAgent initialization:

```python
# In process_chat_request method, after PolicyAgent creation
try:
    agents["context_builder"] = agent_factory.create_context_builder_agent(
        enable_optimization=False
    )
    print(f"ContextBuilderAgent initialized")
except Exception as e:
    print(f"Warning: Could not initialize ContextBuilderAgent: {str(e)}")
```

---

## Prompt Template

**File**: `prompts/ContextBuilderAgent/v1.txt`

```
You are a Context Builder Agent. Your role is to structure and optimize context for response generation.

INPUT CONTEXT:
{{context}}

TASK:
1. Review the provided context bundle
2. Identify any redundant or low-relevance information
3. Prioritize information that will be most useful for generating a response
4. Suggest optimizations (if needed)

CONTEXT OPTIMIZATION RULES:
- Keep all user intent and sentiment data (critical for response tone)
- Keep passages with relevance score >= 0.5
- For conversations longer than 10 turns, summarize older turns
- Preserve all policy constraints and allowed actions
- Maintain all extracted entities

OUTPUT FORMAT:
Provide a brief optimization summary if changes are needed, otherwise respond with "CONTEXT_OPTIMAL".
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/test_context_builder_agent.py`

```python
"""
Unit tests for ContextBuilderAgent
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from agents.ContextBuilderAgent import (
    ContextBuilderAgent,
    ContextBundle,
    UserQueryContext,
    UserContext,
    KnowledgeContext,
    PolicyContext,
    ConversationContext
)
from models.state import ConversationState


@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    client = Mock()
    client.generate = AsyncMock(return_value="CONTEXT_OPTIMAL")
    return client


@pytest.fixture
def mock_prompt_loader():
    """Mock prompt loader"""
    loader = Mock()
    loader.load_prompt = Mock(return_value="Test prompt")
    return loader


@pytest.fixture
def context_builder_agent(mock_llm_client, mock_prompt_loader):
    """Create ContextBuilderAgent for testing"""
    return ContextBuilderAgent(
        llm_client=mock_llm_client,
        prompt_loader=mock_prompt_loader,
        enable_optimization=False
    )


@pytest.fixture
def sample_state():
    """Create sample conversation state"""
    return ConversationState(
        tenant_id="default",
        chatbot_id="test_bot",
        session_id="test_session",
        user_message="I want a refund for my order",
        intent={
            "intent": "refund_request",
            "sub_intent": "full_refund",
            "confidence_score": 0.92
        },
        sentiment={
            "sentiment": "neutral",
            "urgency_score": 0.6,
            "toxicity_flag": False
        },
        rag_results={
            "retrieved_passages": [
                "Refunds are processed within 5-7 business days.",
                "Customer must provide order number for refund."
            ],
            "source_ids": ["doc1", "doc2"],
            "relevance_scores": [0.85, 0.78]
        },
        policy_results={
            "allowed_actions": ["process_refund", "ask_order_number"],
            "denied_actions": ["immediate_refund"],
            "conditions": ["order_verified", "within_return_window"]
        },
        entities={
            "order_id": "ORD-12345"
        },
        conversation_history=[
            {"user": "Hi", "assistant": "Hello! How can I help?"},
            {"user": "I need help with an order", "assistant": "Sure, what's the issue?"}
        ]
    )


class TestContextBuilderAgent:
    """Test suite for ContextBuilderAgent"""

    @pytest.mark.asyncio
    async def test_process_creates_context_bundle(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that process creates a valid context bundle"""
        state_update = await context_builder_agent.process(sample_state)

        assert state_update.context_bundle is not None
        assert "user_query" in state_update.context_bundle
        assert "user_context" in state_update.context_bundle
        assert "knowledge" in state_update.context_bundle
        assert "policy" in state_update.context_bundle
        assert "conversation" in state_update.context_bundle

    @pytest.mark.asyncio
    async def test_user_query_context_structure(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that user query context is correctly structured"""
        state_update = await context_builder_agent.process(sample_state)
        user_query = state_update.context_bundle["user_query"]

        assert user_query["original_message"] == "I want a refund for my order"
        assert user_query["intent"] == "refund_request"
        assert user_query["confidence"] == 0.92
        assert "order_id" in user_query["entities"]

    @pytest.mark.asyncio
    async def test_user_context_structure(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that user context is correctly structured"""
        state_update = await context_builder_agent.process(sample_state)
        user_context = state_update.context_bundle["user_context"]

        assert user_context["sentiment"] == "neutral"
        assert user_context["urgency"] == 0.6
        assert user_context["toxicity_flag"] is False

    @pytest.mark.asyncio
    async def test_knowledge_context_structure(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that knowledge context is correctly structured"""
        state_update = await context_builder_agent.process(sample_state)
        knowledge = state_update.context_bundle["knowledge"]

        assert len(knowledge["retrieved_passages"]) == 2
        assert knowledge["total_passages"] == 2
        assert knowledge["source_ids"] == ["doc1", "doc2"]
        assert knowledge["relevance_scores"] == [0.85, 0.78]

    @pytest.mark.asyncio
    async def test_policy_context_structure(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that policy context is correctly structured"""
        state_update = await context_builder_agent.process(sample_state)
        policy = state_update.context_bundle["policy"]

        assert "process_refund" in policy["allowed_actions"]
        assert "immediate_refund" in policy["denied_actions"]
        assert "order_verified" in policy["conditions"]

    @pytest.mark.asyncio
    async def test_conversation_context_structure(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that conversation context is correctly structured"""
        state_update = await context_builder_agent.process(sample_state)
        conversation = state_update.context_bundle["conversation"]

        assert conversation["turn_count"] == 2
        assert len(conversation["history"]) == 2

    @pytest.mark.asyncio
    async def test_conversation_summary_for_long_history(
        self,
        context_builder_agent
    ):
        """Test that long conversations are summarized"""
        # Create state with long conversation history
        long_history = [
            {"user": f"Message {i}", "assistant": f"Response {i}"}
            for i in range(15)
        ]

        state = ConversationState(
            tenant_id="default",
            chatbot_id="test_bot",
            session_id="test_session",
            user_message="Latest message",
            conversation_history=long_history
        )

        state_update = await context_builder_agent.process(state)
        conversation = state_update.context_bundle["conversation"]

        assert conversation["turn_count"] == 15
        assert conversation["summary"] is not None

    @pytest.mark.asyncio
    async def test_metadata_in_context_bundle(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that metadata is correctly added"""
        state_update = await context_builder_agent.process(sample_state)
        metadata = state_update.context_bundle["metadata"]

        assert metadata["session_id"] == "test_session"
        assert metadata["chatbot_id"] == "test_bot"
        assert "agent_execution_order" in metadata

    @pytest.mark.asyncio
    async def test_tenant_id_preserved(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that tenant_id is preserved in context bundle"""
        state_update = await context_builder_agent.process(sample_state)
        assert state_update.context_bundle["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_timestamp_present(
        self,
        context_builder_agent,
        sample_state
    ):
        """Test that timestamp is added to context bundle"""
        state_update = await context_builder_agent.process(sample_state)
        assert "timestamp" in state_update.context_bundle
        assert "T" in state_update.context_bundle["timestamp"]  # ISO format

    @pytest.mark.asyncio
    async def test_handles_missing_intent_gracefully(
        self,
        context_builder_agent
    ):
        """Test that missing intent is handled gracefully"""
        state = ConversationState(
            tenant_id="default",
            chatbot_id="test_bot",
            session_id="test_session",
            user_message="Test message",
            intent=None
        )

        state_update = await context_builder_agent.process(state)
        user_query = state_update.context_bundle["user_query"]

        assert user_query["intent"] == "unknown"
        assert user_query["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_handles_empty_rag_results(
        self,
        context_builder_agent
    ):
        """Test that empty RAG results are handled"""
        state = ConversationState(
            tenant_id="default",
            chatbot_id="test_bot",
            session_id="test_session",
            user_message="Test message",
            rag_results=None
        )

        state_update = await context_builder_agent.process(state)
        knowledge = state_update.context_bundle["knowledge"]

        assert knowledge["total_passages"] == 0
        assert knowledge["retrieved_passages"] == []


class TestContextBundleModels:
    """Test suite for ContextBundle Pydantic models"""

    def test_user_query_context_model(self):
        """Test UserQueryContext validation"""
        context = UserQueryContext(
            original_message="Test message",
            intent="test_intent",
            confidence=0.9,
            entities={"key": "value"}
        )
        assert context.original_message == "Test message"
        assert context.intent == "test_intent"

    def test_context_bundle_model(self):
        """Test complete ContextBundle validation"""
        bundle = ContextBundle(
            user_query=UserQueryContext(
                original_message="Test",
                intent="test",
                confidence=0.8
            ),
            user_context=UserContext(
                sentiment="neutral"
            ),
            knowledge=KnowledgeContext(),
            policy=PolicyContext(),
            conversation=ConversationContext(),
            timestamp="2024-01-01T00:00:00"
        )
        assert bundle.tenant_id == "default"
        assert bundle.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Integration Tests

**File**: `tests/test_context_builder_integration.py`

```python
"""
Integration tests for ContextBuilderAgent in the full pipeline
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from core.Orchestrator import Orchestrator
from models.state import ConversationState
from schemas.chat import ChatRequest


@pytest.mark.asyncio
async def test_context_builder_in_orchestrator_flow():
    """Test ContextBuilderAgent execution within orchestrator"""
    # This test verifies that ContextBuilderAgent executes
    # in the correct position within the orchestrator flow

    # Create mock agents
    mock_intent_agent = Mock()
    mock_intent_agent.process = AsyncMock(
        return_value=Mock(intent={"intent": "test", "confidence_score": 0.9})
    )

    mock_sentiment_agent = Mock()
    mock_sentiment_agent.process = AsyncMock(
        return_value=Mock(sentiment={"sentiment": "neutral"})
    )

    mock_rag_agent = Mock()
    mock_rag_agent.process = AsyncMock(
        return_value=Mock(
            rag_results={
                "retrieved_passages": ["Test passage"],
                "relevance_scores": [0.8]
            }
        )
    )

    mock_policy_agent = Mock()
    mock_policy_agent.process = AsyncMock(
        return_value=Mock(
            policy_results={
                "allowed_actions": ["test_action"]
            }
        )
    )

    mock_context_builder = Mock()
    mock_context_builder.process = AsyncMock(
        return_value=Mock(
            context_bundle={
                "user_query": {"intent": "test"},
                "user_context": {"sentiment": "neutral"}
            }
        )
    )

    agents = {
        "intent": mock_intent_agent,
        "sentiment": mock_sentiment_agent,
        "rag": mock_rag_agent,
        "policy": mock_policy_agent,
        "context_builder": mock_context_builder
    }

    orchestrator = Orchestrator(agents=agents, config={})

    request = ChatRequest(
        tenant_id="default",
        chatbot_id="test",
        session_id="test_session",
        message="Test message"
    )

    state = await orchestrator.process_request(request)

    # Verify context_builder was called
    mock_context_builder.process.assert_called_once()

    # Verify context_bundle exists in final state
    assert state.context_bundle is not None
```

---

## Verification Checklist

### Agent Implementation

- [ ] ContextBuilderAgent class created in `agents/ContextBuilderAgent.py`
- [ ] All context models defined (UserQueryContext, UserContext, etc.)
- [ ] `process()` method implements all three phases (collection, structuring, optimization)
- [ ] State updates only modify `context_bundle` field
- [ ] Handles missing data gracefully (None checks)
- [ ] Tenant ID preserved in context bundle
- [ ] Timestamp added in ISO format

### Prompt Template

- [ ] Prompt template created at `prompts/ContextBuilderAgent/v1.txt`
- [ ] Prompt includes context optimization rules
- [ ] Prompt specifies output format

### Orchestrator Integration

- [ ] AgentFactory has `create_context_builder_agent()` method
- [ ] Orchestrator execution order includes "context_builder"
- [ ] Execution graph has edges: policy → context_builder → response
- [ ] ChatService initializes ContextBuilderAgent
- [ ] ContextBuilderAgent added after PolicyAgent in ChatService

### Testing

- [ ] Unit tests created in `tests/test_context_builder_agent.py`
- [ ] All unit tests pass
- [ ] Integration test created in `tests/test_context_builder_integration.py`
- [ ] Integration test passes
- [ ] Manual testing with real chat requests

### Documentation

- [ ] Implementation plan documented
- [ ] API documentation updated
- [ ] Architecture diagram updated
- [ ] Integration status updated in AGENT_INTEGRATION_STATUS.md

---

## Files to Create

| File Path | Purpose |
|-----------|---------|
| `agents/ContextBuilderAgent.py` | Main agent implementation |
| `prompts/ContextBuilderAgent/v1.txt` | Prompt template |
| `tests/test_context_builder_agent.py` | Unit tests |
| `tests/test_context_builder_integration.py` | Integration tests |

## Files to Modify

| File Path | Changes |
|-----------|---------|
| `core/AgentFactory.py` | Add `create_context_builder_agent()` method |
| `core/Orchestrator.py` | Update execution order and graph |
| `services/ChatService.py` | Initialize ContextBuilderAgent |
| `docs/AGENT_INTEGRATION_STATUS.md` | Update status tracker |

---

## Expected Outcomes

### 1. Successful Context Building

When ContextBuilderAgent processes a state with all previous agent outputs:

```python
# Input
state = ConversationState(
    user_message="I want to return my order #12345",
    intent={"intent": "return_request", "confidence_score": 0.95},
    sentiment={"sentiment": "frustrated", "urgency_score": 0.7},
    rag_results={
        "retrieved_passages": [
            "Returns must be initiated within 30 days of delivery.",
            "Order #12345 was delivered on 2024-01-15."
        ],
        "source_ids": ["policy_doc_1", "orders_db"],
        "relevance_scores": [0.92, 0.88]
    },
    policy_results={
        "allowed_actions": ["process_return", "check_eligibility"],
        "denied_actions": ["instant_refund"],
        "conditions": ["within_return_window", "item_undamaged"]
    },
    entities={"order_id": "12345"},
    conversation_history=[...]
)

# Output
context_bundle = {
    "user_query": {
        "original_message": "I want to return my order #12345",
        "entities": {"order_id": "12345"},
        "intent": "return_request",
        "confidence": 0.95
    },
    "user_context": {
        "sentiment": "frustrated",
        "urgency": 0.7,
        "toxicity_flag": False,
        "profile": None
    },
    "knowledge": {
        "retrieved_passages": [
            "Returns must be initiated within 30 days of delivery.",
            "Order #12345 was delivered on 2024-01-15."
        ],
        "source_ids": ["policy_doc_1", "orders_db"],
        "relevance_scores": [0.92, 0.88],
        "total_passages": 2
    },
    "policy": {
        "allowed_actions": ["process_return", "check_eligibility"],
        "denied_actions": ["instant_refund"],
        "conditions": ["within_return_window", "item_undamaged"],
        "reasoning": None
    },
    "conversation": {
        "history": [...],
        "turn_count": 3,
        "summary": None
    },
    "metadata": {
        "session_id": "abc123",
        "chatbot_id": "bot_1",
        "agent_execution_order": ["intent", "sentiment", "rag", "policy"]
    },
    "tenant_id": "default",
    "timestamp": "2024-01-20T14:30:45.123456"
}
```

### 2. Graceful Handling of Missing Data

When some agents haven't produced output:

```python
# Input - no RAG results
state = ConversationState(
    user_message="Hello",
    intent={"intent": "greeting", "confidence_score": 0.98},
    sentiment={"sentiment": "positive", "urgency_score": 0.2},
    rag_results=None,  # Missing
    policy_results=None,  # Missing
    entities={},
    conversation_history=[]
)

# Output - still creates valid bundle
context_bundle = {
    "user_query": {...},
    "user_context": {...},
    "knowledge": {
        "retrieved_passages": [],
        "source_ids": [],
        "relevance_scores": [],
        "total_passages": 0
    },
    "policy": {
        "allowed_actions": [],
        "denied_actions": [],
        "conditions": [],
        "reasoning": None
    },
    "conversation": {
        "history": [],
        "turn_count": 0,
        "summary": None
    },
    ...
}
```

### 3. Long Conversation Summarization

When conversation history exceeds 10 turns:

```python
# Input - 15 conversation turns
state.conversation_history = [turn for turn in range(15)]

# Output - summary created
context_bundle["conversation"] = {
    "history": [...],  # Full history preserved
    "turn_count": 15,
    "summary": "Conversation covers: refund discussion, order inquiry"
}
```

---

## Troubleshooting

### Issue: Context bundle not appearing in final state

**Symptoms**: ResolutionAgent doesn't have access to context_bundle

**Solutions**:
1. Verify ContextBuilderAgent is added to agents dict in ChatService
2. Check orchestrator execution order includes "context_builder"
3. Ensure execution graph has proper edges (policy → context_builder → response)
4. Check that StateUpdate is returned from process() method

### Issue: Missing fields in context bundle

**Symptoms**: Some fields in context_bundle are None or empty

**Solutions**:
1. Verify previous agents (Intent, Sentiment, RAG, Policy) are executing successfully
2. Check that state has the expected fields before ContextBuilderAgent processes
3. Add logging to see what data is available in state
4. Ensure None checks are working properly in _build_*_context methods

### Issue: Pydantic validation errors

**Symptoms**: Validation errors when creating ContextBundle

**Solutions**:
1. Check all required fields are present
2. Verify field types match expectations
3. Add default values in Pydantic models
4. Use Field(default=...) for optional fields

---

## Next Steps

After ContextBuilderAgent implementation:

1. **Update ResponseAgent**: Modify ResolutionAgent to use `context_bundle` instead of individual state fields
2. **Implement EscalationAgent**: Create the Tier 3 escalation decision agent
3. **Implement TicketActionAgent**: Create the Tier 3 ticket action agent
4. **Performance Testing**: Measure context building overhead
5. **Optimization Enhancement**: Implement LLM-based context optimization if needed

---

## References

- [Agent Specification Document](../chatgpt-documentations/Agents%20Required.md)
- [Architecture Documentation](./CUSTOMER_SUPPORT_RESOLUTION_ARCHITECTURE.md)
- [PolicyAgent Implementation Plan](./POLICY_AGENT_IMPLEMENTATION_PLAN.md)
- [Integration Status Tracker](./AGENT_INTEGRATION_STATUS.md)
