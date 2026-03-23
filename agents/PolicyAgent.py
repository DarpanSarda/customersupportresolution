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
        print(f"[DEBUG] PolicyAgent initialized with config_service={config_service is not None}")

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a policy evaluation specialist for a customer support system. Your role is to analyze customer requests against business policies and recommend appropriate actions.

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

Remember: Your recommendations guide the next steps in handling the customer request. Be thorough and accurate."""

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
        print(f"[DEBUG] PolicyAgent context_bundle: {context_bundle}")

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
        print(f"[DEBUG] Policy details: required_fields={policy.get('required_fields')}, escalate_threshold={policy.get('escalate_if_sentiment_above')}")

        # Evaluate policy
        evaluation = await self._evaluate_policy(
            policy=policy,
            intent=intent,
            intent_confidence=intent_confidence,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            context_bundle=context_bundle
        )

        print(f"[DEBUG] Policy evaluation complete: action={evaluation.get('business_action')}, escalate={evaluation.get('escalate')}, allow_execution={evaluation.get('allow_execution')}")

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
        """
        Load policy from database via ConfigService.

        Args:
            tenant_id: Tenant identifier
            intent: Intent type

        Returns:
            Policy dict or None if not found
        """
        if not intent:
            print(f"[DEBUG] No intent provided, cannot load policy")
            return None

        if not self.config_service:
            print(f"[DEBUG] No config_service configured for PolicyAgent")
            return None

        # Check cache first
        cache_key = f"{tenant_id}:{intent}"
        if cache_key in self._policy_cache:
            print(f"[DEBUG] Policy cache HIT for {cache_key}")
            return self._policy_cache[cache_key]

        print(f"[DEBUG] Policy cache MISS for {cache_key}, loading from DB")

        # Load from database
        try:
            policy = await self.config_service.get_policy(tenant_id, intent)
            if policy:
                self._policy_cache[cache_key] = policy
                print(f"[DEBUG] Loaded policy from DB: {cache_key}, action={policy.get('business_action')}")
            else:
                print(f"[DEBUG] No policy found in DB: {cache_key}")
            return policy
        except Exception as e:
            print(f"[DEBUG] Error loading policy: {str(e)}")
            import traceback
            traceback.print_exc()
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
        """
        Evaluate policy against current state.

        Args:
            policy: Policy configuration
            intent: Detected intent
            intent_confidence: Intent confidence score
            sentiment: Detected sentiment
            sentiment_confidence: Sentiment confidence/urgency score
            context_bundle: Context with extracted entities

        Returns:
            Evaluation results dict
        """
        print(f"[DEBUG] Evaluating policy: {policy.get('business_action')}")

        # Extract policy rules
        business_action = policy.get("business_action")
        required_fields = policy.get("required_fields", [])
        escalate_threshold = policy.get("escalate_if_sentiment_above")
        blocked_sentiments = policy.get("blocked_sentiments", [])
        priority = policy.get("priority", "normal")

        print(f"[DEBUG] Policy rules - required_fields={required_fields}, blocked_sentiments={blocked_sentiments}, escalate_threshold={escalate_threshold}")

        # Check for blocked sentiments
        if sentiment and blocked_sentiments:
            sentiment_lower = sentiment.lower()
            blocked_lower = [s.lower() for s in blocked_sentiments]
            if sentiment_lower in blocked_lower:
                print(f"[DEBUG] Blocked sentiment detected: {sentiment} is in {blocked_sentiments}")
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
        if escalate_threshold is not None and sentiment_confidence >= escalate_threshold:
            escalate = True
            priority = "high"
            print(f"[DEBUG] Escalation triggered: sentiment_confidence={sentiment_confidence} >= threshold={escalate_threshold}")

        # Check required fields
        missing_fields = []
        if required_fields:
            for field in required_fields:
                if field not in context_bundle:
                    missing_fields.append(field)

        if missing_fields:
            print(f"[DEBUG] Missing required fields: {missing_fields}")

        # Determine if execution should proceed
        allow_execution = not missing_fields and not escalate

        print(f"[DEBUG] Evaluation result - allow_execution={allow_execution}, escalate={escalate}, priority={priority}")

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
        """
        Build human-readable explanation for policy decision.

        Args:
            intent: Detected intent
            sentiment: Detected sentiment
            missing_fields: List of missing required fields
            escalate: Whether escalation was triggered
            business_action: Recommended business action

        Returns:
            Human-readable reason string
        """
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
        """
        Get agent information.

        Returns:
            Dictionary with agent details
        """
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
