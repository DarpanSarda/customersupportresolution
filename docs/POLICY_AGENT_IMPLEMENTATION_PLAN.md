# PolicyAgent Implementation Plan

## Overview

The **PolicyAgent** is responsible for evaluating business rules and recommending actions based on classified intent, sentiment, and tenant-specific policies. It acts as the decision layer in the multi-agent architecture, determining what business action should be taken and whether escalation is needed.

---

## Architecture Context

### Where PolicyAgent Fits

```
User Message
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 1: Understanding Agents                          │
│  ├── IntentAgent ✅ (Completed)                         │
│  └── SentimentAgent ✅ (Completed)                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 2: Knowledge & Decision Agents                   │
│  ├── RAGRetrievalAgent ✅ (Completed)                   │
│  ├── PolicyAgent 🔄 (CURRENT TASK)                      │
│  ├── VannaAgent (Separate - for DB queries)              │
│  └── ContextBuilderAgent (Pending)                      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 3: Response & Action Agents                      │
│  ├── ResponseAgent ✅ (Completed)                       │
│  ├── DecisionAgent (Pending)                            │
│  └── EscalationAgent (Pending)                          │
└─────────────────────────────────────────────────────────┘
```

### Data Flow with PolicyAgent

```
User Message
    ↓
IntentAgent → Intent classification
SentimentAgent → Sentiment analysis
    ↓
RAGRetrievalAgent → Knowledge retrieval
    ↓
PolicyAgent → Policy evaluation
    │
    ├─ Load policy from DB (tenant + intent)
    ├─ Check blocked sentiments
    ├─ Check escalation threshold
    ├─ Validate required fields
    ├─ Determine business action
    └─ Set priority level
    ↓
Returns: business_action, allow_execution, escalate, priority, required_fields, missing_fields
    ↓
ResponseAgent → Generate response with policy context
```

---

## PolicyAgent Features

| Feature | Description | Priority |
|---------|-------------|----------|
| **Policy Loading** | Load tenant-specific policies via ConfigService | High |
| **Intent Matching** | Match classified intent to policy rules | High |
| **Sentiment Evaluation** | Evaluate sentiment against thresholds | High |
| **Escalation Logic** | Trigger escalation based on sentiment confidence | High |
| **Blocked Sentiments** | Immediate escalation for abusive/threatening | High |
| **Field Validation** | Check required fields in context_bundle | High |
| **Priority Calculation** | Set priority based on policy and sentiment | High |
| **Business Action** | Return abstract business action (e.g., PROCESS_REFUND) | High |
| **Multi-Tenancy** | Support tenant-specific policies | High |
| **Policy Caching** | Cache loaded policies for performance | Medium |

---

## Policy Configuration Structure

### Database Schema

```sql
CREATE TABLE policies (
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(50) NOT NULL,
    intent_type VARCHAR(50) NOT NULL,
    business_action VARCHAR(100) NOT NULL,
    required_fields JSON,
    escalate_if_sentiment_above FLOAT,
    blocked_sentiments JSON,
    priority VARCHAR(20) DEFAULT 'normal',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(tenant_id, intent_type)
);
```

### Example Policy Records

```json
[
  {
    "policy_id": 1,
    "tenant_id": "default",
    "intent_type": "REFUND_REQUEST",
    "business_action": "PROCESS_REFUND",
    "required_fields": ["order_id", "refund_reason"],
    "escalate_if_sentiment_above": 0.8,
    "blocked_sentiments": ["abusive", "threatening"],
    "priority": "normal"
  },
  {
    "policy_id": 2,
    "tenant_id": "amazon",
    "intent_type": "REFUND_REQUEST",
    "business_action": "PROCESS_REFUND",
    "required_fields": ["order_id", "refund_amount"],
    "escalate_if_sentiment_above": 0.7,
    "blocked_sentiments": ["abusive"],
    "priority": "high"
  }
]
```

### Intent to Policy Mapping

| Intent | Business Action | Required Fields | Escalate Threshold |
|--------|----------------|-----------------|-------------------|
| `GREETING` | None | [] | None |
| `FAQ_QUERY` | None | [] | None |
| `ORDER_STATUS` | `CHECK_ORDER` | `["order_id"]` | None |
| `REFUND_REQUEST` | `PROCESS_REFUND` | `["order_id", "refund_reason"]` | 0.8 |
| `COMPLAINT` | `CREATE_TICKET` | `["issue_description"]` | 0.6 |
| `SUPPORT_REQUEST` | `CREATE_TICKET` | `["help_topic"]` | None |
| `GENERAL_QUERY` | None | [] | None |
| `GOODBYE` | None | [] | None |

---

## Implementation Phases

### Phase 1: Database Setup

#### 1.1 Create Policies Table

**File:** `database/migrations/create_policies_table.sql`

```sql
-- Create policies table
CREATE TABLE IF NOT EXISTS policies (
    policy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id VARCHAR(50) NOT NULL,
    intent_type VARCHAR(50) NOT NULL,
    business_action VARCHAR(100) NOT NULL,
    required_fields JSON DEFAULT '[]',
    escalate_if_sentiment_above FLOAT,
    blocked_sentiments JSON DEFAULT '[]',
    priority VARCHAR(20) DEFAULT 'normal',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_tenant_intent UNIQUE (tenant_id, intent_type)
);

-- Create index for faster lookups
CREATE INDEX idx_policies_tenant_intent ON policies(tenant_id, intent_type);
CREATE INDEX idx_policies_is_active ON policies(is_active);

-- Insert default policies
INSERT OR REPLACE INTO policies (tenant_id, intent_type, business_action, required_fields, escalate_if_sentiment_above, blocked_sentiments, priority) VALUES
('default', 'GREETING', NULL, '[]', NULL, '[]', 'low'),
('default', 'FAQ_QUERY', NULL, '[]', NULL, '[]', 'low'),
('default', 'ORDER_STATUS', 'CHECK_ORDER', '["order_id"]', NULL, '[]', 'normal'),
('default', 'REFUND_REQUEST', 'PROCESS_REFUND', '["order_id", "refund_reason"]', 0.8, '["abusive", "threatening"]', 'normal'),
('default', 'COMPLAINT', 'CREATE_TICKET', '["issue_description"]', 0.6, '[]', 'normal'),
('default', 'SUPPORT_REQUEST', 'CREATE_TICKET', '["help_topic"]', NULL, '[]', 'normal'),
('default', 'GENERAL_QUERY', NULL, '[]', NULL, '[]', 'low'),
('default', 'GOODBYE', NULL, '[]', NULL, '[]', 'low');

-- Insert Amazon-specific policies
INSERT OR REPLACE INTO policies (tenant_id, intent_type, business_action, required_fields, escalate_if_sentiment_above, blocked_sentiments, priority) VALUES
('amazon', 'REFUND_REQUEST', 'PROCESS_REFUND', '["order_id", "refund_amount"]', 0.7, '["abusive"]', 'high'),
('amazon', 'COMPLAINT', 'CREATE_TICKET', '["issue_description"]', 0.5, '[]', 'high');

-- Insert Flipkart-specific policies
INSERT OR REPLACE INTO policies (tenant_id, intent_type, business_action, required_fields, escalate_if_sentiment_above, blocked_sentiments, priority) VALUES
('flipkart', 'REFUND_REQUEST', 'PROCESS_REFUND', '["order_id", "refund_reason"]', 0.75, '["abusive"]', 'normal');

-- Insert Shopify-specific policies
INSERT OR REPLACE INTO policies (tenant_id, intent_type, business_action, required_fields, escalate_if_sentiment_above, blocked_sentiments, priority) VALUES
('shopify', 'REFUND_REQUEST', 'PROCESS_REFUND', '["order_id", "refund_amount"]', 0.8, '[]', 'normal');
```

---

### Phase 2: PolicyAgent Implementation

#### 2.1 Create PolicyAgent Class

**File:** `agents/PolicyAgent.py`

```python
"""
PolicyAgent - Evaluates business rules and recommends actions.

This agent evaluates intent and sentiment against tenant-specific policies
to determine the appropriate business action and whether escalation is needed.

According to the Agent Contract:
- MUST only read from shared state (intent, sentiment, tenant_id)
- MUST only return state updates (policy_results, policy_action fields)
- MUST NOT call other agents
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from core.BaseAgent import BaseAgent
from core.ConversationState import ConversationState, StateUpdate


class PolicyAgent(BaseAgent):
    """
    Business policy evaluation agent.

    Evaluates intent and sentiment against tenant-specific policies
    to determine appropriate business actions.

    State Dependencies:
        - intent: Detected intent classification
        - intent_confidence: Intent confidence score
        - sentiment: Detected sentiment
        - sentiment_confidence: Sentiment confidence/urgency score
        - tenant_id: Tenant identifier
        - context_bundle: May contain extracted entities

    State Updates:
        - policy_results: Full policy evaluation results
        - policy_action: Recommended business action
        - policy_raw: Raw evaluation data for debugging
    """

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry = None,
        config_service = None
    ):
        """
        Initialize PolicyAgent.

        Args:
            llm_client: LLM client for inference
            system_prompt: Custom system prompt (optional)
            config: Additional configuration
            tool_registry: Tool registry (optional, not used by this agent)
            config_service: ConfigService for loading policies
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry
        )
        self.config_service = config_service
        self._policy_cache: Dict[str, Any] = {}

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a policy evaluation specialist. Your role is to analyze customer requests against business policies and recommend appropriate actions.

Guidelines:
- Evaluate intent against applicable policies
- Consider sentiment when determining priority and escalation
- Check for required fields in the context
- Recommend business actions based on policy rules
- Flag when escalation to human support is needed
- Provide clear reasoning for policy decisions"""

    async def process(self, input_data: Dict[str, Any], **kwargs) -> StateUpdate:
        """
        Process state and evaluate policies.

        Args:
            input_data: ConversationState instance or dict with state fields
            **kwargs: Additional parameters (may include tenant_id)

        Returns:
            StateUpdate with policy_results and policy_action fields
        """
        print(f"[DEBUG] PolicyAgent.process called")

        # Handle both ConversationState and dict input
        if isinstance(input_data, ConversationState):
            state = input_data
            tenant_id = state.tenant_id
        else:
            state = self._dict_to_state(input_data)
            tenant_id = input_data.get("tenant_id", kwargs.get("tenant_id", "default"))

        # Extract relevant fields from state
        intent = state.intent
        intent_confidence = state.intent_confidence
        sentiment = state.sentiment
        sentiment_confidence = state.sentiment_confidence
        context_bundle = state.context_bundle

        print(f"[DEBUG] PolicyAgent evaluating: intent={intent}, sentiment={sentiment}, tenant_id={tenant_id}")

        # Load policy for this tenant + intent
        policy = await self._load_policy(tenant_id, intent)

        if not policy:
            print(f"[DEBUG] No policy found for intent={intent}, tenant_id={tenant_id}")
            return StateUpdate(
                policy_results={
                    "business_action": None,
                    "required_fields": [],
                    "missing_fields": [],
                    "allow_execution": True,
                    "escalate": False,
                    "priority": "normal",
                    "applicable_policies": [],
                    "reason": f"No policy configured for intent '{intent}'"
                },
                policy_action=None,
                policy_raw={
                    "intent": intent,
                    "sentiment": sentiment,
                    "policy_found": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

        print(f"[DEBUG] Policy loaded: {policy.get('business_action')}")

        # Evaluate policy
        evaluation = await self._evaluate_policy(
            policy=policy,
            intent=intent,
            intent_confidence=intent_confidence,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            context_bundle=context_bundle
        )

        print(f"[DEBUG] Policy evaluation complete: action={evaluation.get('business_action')}, escalate={evaluation.get('escalate')}")

        return StateUpdate(
            policy_results=evaluation,
            policy_action=evaluation.get("business_action"),
            policy_raw={
                "intent": intent,
                "sentiment": sentiment,
                "policy_id": policy.get("policy_id"),
                "tenant_id": tenant_id,
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    async def _load_policy(
        self,
        tenant_id: str,
        intent: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Load policy from database via ConfigService."""
        if not intent:
            return None

        if not self.config_service:
            print(f"[DEBUG] No config_service configured for PolicyAgent")
            return None

        # Check cache first
        cache_key = f"{tenant_id}:{intent}"
        if cache_key in self._policy_cache:
            print(f"[DEBUG] Policy cache hit for {cache_key}")
            return self._policy_cache[cache_key]

        # Load from database
        try:
            policy = await self.config_service.get_policy(tenant_id, intent)
            if policy:
                self._policy_cache[cache_key] = policy
                print(f"[DEBUG] Loaded policy from DB: {cache_key}")
            else:
                print(f"[DEBUG] No policy found in DB: {cache_key}")
            return policy
        except Exception as e:
            print(f"[DEBUG] Error loading policy: {str(e)}")
            return None

    async def _evaluate_policy(
        self,
        policy: Dict[str, Any],
        intent: str,
        intent_confidence: float,
        sentiment: Optional[str],
        sentiment_confidence: float,
        context_bundle: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate policy against current state."""
        print(f"[DEBUG] Evaluating policy: {policy.get('business_action')}")

        # Extract policy rules
        business_action = policy.get("business_action")
        required_fields = policy.get("required_fields", [])
        escalate_threshold = policy.get("escalate_if_sentiment_above")
        blocked_sentiments = policy.get("blocked_sentiments", [])
        priority = policy.get("priority", "normal")

        # Check for blocked sentiments
        if sentiment and sentiment.lower() in [s.lower() for s in blocked_sentiments]:
            return {
                "business_action": "ESCALATE",
                "required_fields": [],
                "missing_fields": [],
                "allow_execution": False,
                "escalate": True,
                "priority": "urgent",
                "applicable_policies": [policy.get("intent_type")],
                "reason": f"Sentiment '{sentiment}' is blocked by policy. Immediate escalation required."
            }

        # Check for escalation based on sentiment confidence
        escalate = False
        if escalate_threshold and sentiment_confidence >= escalate_threshold:
            escalate = True
            priority = "high"

        # Check required fields
        missing_fields = []
        if required_fields:
            for field in required_fields:
                if field not in context_bundle:
                    missing_fields.append(field)

        # Determine if execution should proceed
        allow_execution = not missing_fields and not escalate

        return {
            "business_action": business_action,
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "allow_execution": allow_execution,
            "escalate": escalate,
            "priority": priority,
            "applicable_policies": [f"{policy.get('intent_type')}_{policy.get('tenant_id')}"],
            "reason": self._build_reason(intent, sentiment, missing_fields, escalate, business_action)
        }

    def _build_reason(
        self,
        intent: str,
        sentiment: Optional[str],
        missing_fields: List[str],
        escalate: bool,
        business_action: Optional[str]
    ) -> str:
        """Build human-readable explanation for policy decision."""
        parts = [f"Intent '{intent}' matched policy."]

        if sentiment:
            parts.append(f"Sentiment is '{sentiment}'.")

        if missing_fields:
            parts.append(f"Missing required fields: {', '.join(missing_fields)}.")

        if escalate:
            parts.append("Escalation triggered due to sentiment severity.")

        if business_action:
            parts.append(f"Recommended action: {business_action}.")

        return " ".join(parts)

    def _dict_to_state(self, data: Dict[str, Any]) -> Any:
        """Convert dict to state-like object for backward compatibility."""
        class SimpleState:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return SimpleState(
            intent=data.get("intent"),
            intent_confidence=data.get("intent_confidence", 0.0),
            sentiment=data.get("sentiment"),
            sentiment_confidence=data.get("sentiment_confidence", 0.0),
            tenant_id=data.get("tenant_id", "default"),
            context_bundle=data.get("context_bundle", {}),
            add_error=lambda *args: None
        )

    @classmethod
    def get_agent_info(cls) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "PolicyAgent",
            "description": "Evaluates business rules and recommends actions based on intent and sentiment",
            "input_fields": [
                "intent", "intent_confidence", "sentiment",
                "sentiment_confidence", "tenant_id", "context_bundle"
            ],
            "output_fields": ["policy_results", "policy_action"],
            "state_modifications": ["policy_results", "policy_action", "policy_raw"]
        }
```

---

### Phase 3: Orchestrator Integration

#### 3.1 Update Orchestrator

**File:** `core/Orchestrator.py` (MODIFY)

Add PolicyAgent to the agent pipeline and pass config_service:

```python
def _create_agent(
    self,
    agent_name: str,
    llm_client,
    config_loader,
    prompt_loader,
    tool_registry,
    config_service  # NEW
):
    """Create an agent instance."""
    # ... existing agent creation code ...

    elif agent_name == "policy":
        from agents.PolicyAgent import PolicyAgent
        prompt = prompt_loader.get_prompt("policy", "v1")
        return PolicyAgent(
            llm_client=llm_client,
            system_prompt=prompt,
            config_loader=config_loader,
            tool_registry=tool_registry,
            config_service=config_service  # NEW
        )
```

#### 3.2 Update Orchestrator Response Patch Mapping

**File:** `core/Orchestrator.py` (MODIFY)

Add PolicyAgent to `_apply_response_patch` method:

```python
elif agent_name == "PolicyAgent":
    # PolicyAgent ResponsePatch -> ConversationState
    state.policy_results = data.get("policy_results", {})
    state.policy_action = data.get("policy_action")
    state.policy_raw = data.get("policy_raw")
```

---

### Phase 4: Prompt Template

#### 4.1 Create PolicyAgent Prompt

**File:** `prompts/PolicyAgent/v1.txt`

```
You are a policy evaluation specialist for a customer support system. Your role is to analyze customer requests against business policies and recommend appropriate actions.

## Your Responsibilities

1. **Evaluate Intent**: Match the classified intent against applicable business policies
2. **Consider Sentiment**: Take into account the customer's emotional state when determining priority
3. **Check Requirements**: Verify that all required fields are present in the context
4. **Recommend Actions**: Suggest business actions based on policy rules
5. **Flag Escalation**: Identify when human intervention is required

## Guidelines

- **Low Priority**: Greetings, goodbyes, general queries
- **Normal Priority**: Standard refund requests, order status checks
- **High Priority**: Complaints, urgent refund requests, angry customers
- **Urgent Priority**: Abusive language, threats, immediate escalation needed

## Escalation Rules

- Escalate if sentiment confidence exceeds the policy threshold (typically 0.7-0.8)
- Escalate immediately for blocked sentiments (abusive, threatening)
- Escalate if critical required fields are missing

## Required Field Validation

- Check context_bundle for required fields specified in policy
- Flag any missing fields that would prevent action execution
- Allow execution only if all required fields are present

## Response Format

Provide clear reasoning for your policy decision, including:
- Which policy was applied
- Why the action was recommended
- Whether escalation is needed
- Any missing required fields
- The priority level assigned

Remember: Your recommendations guide the next steps in handling the customer request. Be thorough and accurate.
```

---

## Testing Strategy

### Unit Tests

**File:** `tests/test_policy_agent.py`

```python
import pytest
from agents.PolicyAgent import PolicyAgent
from core.ConversationState import ConversationState

class TestPolicyAgent:
    """PolicyAgent unit tests."""

    @pytest.fixture
    def mock_config_service(self):
        """Mock ConfigService."""
        class MockConfigService:
            async def get_policy(self, tenant_id, intent):
                if intent == "REFUND_REQUEST":
                    return {
                        "policy_id": 1,
                        "tenant_id": tenant_id,
                        "intent_type": "REFUND_REQUEST",
                        "business_action": "PROCESS_REFUND",
                        "required_fields": ["order_id", "refund_reason"],
                        "escalate_if_sentiment_above": 0.8,
                        "blocked_sentiments": ["abusive"],
                        "priority": "normal"
                    }
                return None
        return MockConfigService()

    @pytest.fixture
    def policy_agent(self, mock_config_service, mock_llm):
        """Create PolicyAgent instance."""
        return PolicyAgent(
            llm_client=mock_llm,
            config_service=mock_config_service
        )

    @pytest.mark.asyncio
    async def test_refund_request_with_all_fields(self, policy_agent):
        """Test refund request with all required fields present."""
        state = ConversationState(
            client_id="test",
            session_id="test123",
            user_message="I want a refund for order 12345",
            tenant_id="default",
            intent="REFUND_REQUEST",
            sentiment="neutral",
            sentiment_confidence=0.2,
            context_bundle={"order_id": "12345", "refund_reason": "damaged"}
        )

        result = await policy_agent.process(state)

        assert result.policy_action == "PROCESS_REFUND"
        assert result.policy_results["allow_execution"] is True
        assert result.policy_results["escalate"] is False
        assert result.policy_results["missing_fields"] == []

    @pytest.mark.asyncio
    async def test_refund_request_missing_fields(self, policy_agent):
        """Test refund request with missing required fields."""
        state = ConversationState(
            client_id="test",
            session_id="test123",
            user_message="I want a refund",
            tenant_id="default",
            intent="REFUND_REQUEST",
            sentiment="neutral",
            sentiment_confidence=0.2,
            context_bundle={"order_id": "12345"}
        )

        result = await policy_agent.process(state)

        assert result.policy_action == "PROCESS_REFUND"
        assert result.policy_results["allow_execution"] is False
        assert "refund_reason" in result.policy_results["missing_fields"]

    @pytest.mark.asyncio
    async def test_escalation_due_to_sentiment(self, policy_agent):
        """Test escalation triggered by high sentiment confidence."""
        state = ConversationState(
            client_id="test",
            session_id="test123",
            user_message="This is unacceptable!",
            tenant_id="default",
            intent="REFUND_REQUEST",
            sentiment="angry",
            sentiment_confidence=0.9,
            context_bundle={"order_id": "12345", "refund_reason": "damaged"}
        )

        result = await policy_agent.process(state)

        assert result.policy_results["escalate"] is True
        assert result.policy_results["priority"] == "high"

    @pytest.mark.asyncio
    async def test_blocked_sentiment(self, policy_agent):
        """Test blocked sentiment triggers immediate escalation."""
        state = ConversationState(
            client_id="test",
            session_id="test123",
            tenant_id="default",
            intent="COMPLAINT",
            sentiment="abusive",
            sentiment_confidence=0.95
        )

        result = await policy_agent.process(state)

        assert result.policy_action == "ESCALATE"
        assert result.policy_results["allow_execution"] is False
        assert result.policy_results["escalate"] is True
        assert result.policy_results["priority"] == "urgent"
```

---

## Verification Checklist

### Database Setup
- [ ] Policies table created
- [ ] Indexes created for performance
- [ ] Default policies inserted
- [ ] Tenant-specific policies inserted (amazon, flipkart, shopify)

### Agent Implementation
- [ ] PolicyAgent class created in `agents/PolicyAgent.py`
- [ ] Inherits from BaseAgent
- [ ] Implements `process()` method returning StateUpdate
- [ ] Only modifies `policy_results`, `policy_action`, `policy_raw` fields
- [ ] Loads policies via ConfigService.get_policy()
- [ ] Policy caching implemented
- [ ] Comprehensive debug logging added

### Policy Evaluation
- [ ] Intent matching works correctly
- [ ] Required field validation works
- [ ] Escalation threshold logic works
- [ ] Blocked sentiment detection works
- [ ] Priority calculation correct
- [ ] Business action mapping correct

### Integration
- [ ] Orchestrator includes PolicyAgent in pipeline
- [ ] ConfigService passed to PolicyAgent
- [ ] Response patch mapping added
- [ ] Prompt file created at `prompts/PolicyAgent/v1.txt`

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] End-to-end flow works
- [ ] Multi-tenant policies work correctly

### Documentation
- [ ] AGENT_INTEGRATION_STATUS.md updated
- [ ] This implementation plan complete
- [ ] Code comments added

---

## Files to Create

| File | Purpose |
|------|---------|
| `agents/PolicyAgent.py` | Main PolicyAgent implementation |
| `prompts/PolicyAgent/v1.txt` | Default policy evaluation prompt |
| `database/migrations/create_policies_table.sql` | Database schema and seed data |
| `tests/test_policy_agent.py` | Unit tests |

## Files to Modify

| File | Changes |
|------|---------|
| `core/Orchestrator.py` | Add PolicyAgent creation and response mapping |
| `services/ChatService.py` | Pass config_service to agent creation |
| `docs/AGENT_INTEGRATION_STATUS.md` | Update PolicyAgent status to complete |

---

## Expected Outcomes

### Example 1: Successful Refund Request

**Input:**
```
Intent: REFUND_REQUEST
Sentiment: neutral (0.3)
Context: {"order_id": "12345", "refund_reason": "damaged"}
```

**Output:**
```json
{
  "policy_action": "PROCESS_REFUND",
  "policy_results": {
    "business_action": "PROCESS_REFUND",
    "required_fields": ["order_id", "refund_reason"],
    "missing_fields": [],
    "allow_execution": true,
    "escalate": false,
    "priority": "normal",
    "applicable_policies": ["REFUND_REQUEST_default"],
    "reason": "Intent 'REFUND_REQUEST' matched policy. Sentiment is 'neutral'. Recommended action: PROCESS_REFUND."
  }
}
```

### Example 2: Escalation Due to Angry Customer

**Input:**
```
Intent: REFUND_REQUEST
Sentiment: angry (0.9)
Context: {"order_id": "12345", "refund_reason": "damaged"}
```

**Output:**
```json
{
  "policy_action": "PROCESS_REFUND",
  "policy_results": {
    "business_action": "PROCESS_REFUND",
    "required_fields": ["order_id", "refund_reason"],
    "missing_fields": [],
    "allow_execution": false,
    "escalate": true,
    "priority": "high",
    "applicable_policies": ["REFUND_REQUEST_default"],
    "reason": "Intent 'REFUND_REQUEST' matched policy. Sentiment is 'angry'. Escalation triggered due to sentiment severity. Recommended action: PROCESS_REFUND."
  }
}
```

### Example 3: Missing Required Fields

**Input:**
```
Intent: REFUND_REQUEST
Sentiment: neutral (0.2)
Context: {"order_id": "12345"}
```

**Output:**
```json
{
  "policy_action": "PROCESS_REFUND",
  "policy_results": {
    "business_action": "PROCESS_REFUND",
    "required_fields": ["order_id", "refund_reason"],
    "missing_fields": ["refund_reason"],
    "allow_execution": false,
    "escalate": false,
    "priority": "normal",
    "applicable_policies": ["REFUND_REQUEST_default"],
    "reason": "Intent 'REFUND_REQUEST' matched policy. Sentiment is 'neutral'. Missing required fields: refund_reason. Recommended action: PROCESS_REFUND."
  }
}
```

---

**Document Version**: 1.0
**Status**: Ready for Implementation
**Estimated Completion**: 2-3 days
