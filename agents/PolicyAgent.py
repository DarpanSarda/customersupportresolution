# agents/policy_agent.py

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch
from typing import Dict, List, Any, Optional


class PolicyAgent(BaseAgent):
    """
    Generic policy evaluation agent - NO domain hardcoding.

    Reads intent from state and evaluates against configured policies.
    Outputs abstract business_action for DecisionAgent to route.

    Pure logic - no side effects, no tool calls, no routing.

    Reads:
    - understanding.intent.name: Intent to match against policies
    - understanding.sentiment.confidence: For escalation thresholds
    - context: To check required fields

    Writes:
    - policy.business_action: Abstract action (PROCESS_REFUND, PROCESS_REPLACEMENT, etc.)
    - policy.required_fields: Fields needed for this action
    - policy.missing_fields: Fields not present in context
    - policy.allow_execution: Whether execution should proceed
    - policy.escalate: Whether to escalate to human
    - policy.priority: Priority level (low, normal, high, urgent)
    - policy.reason: Explanation of policy decision

    All policies are config-driven by intent name.
    """

    agent_name = "PolicyAgent"
    allowed_section = "policy"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """
        Evaluate intent against configured policy.

        Returns Patch with abstract business action.
        """
        # -------------------------------------------------
        # 1️⃣ Extract data from state
        # -------------------------------------------------
        understanding = state.get("understanding", {})
        intent_data = understanding.get("intent", {})
        intent_name = intent_data.get("name")

        sentiment_data = understanding.get("sentiment", {})
        sentiment_confidence = sentiment_data.get("confidence", 0.0)
        sentiment_label = sentiment_data.get("label", "NEUTRAL")

        context_data = state.get("context", {})

        # -------------------------------------------------
        # 2️⃣ Get policy config for this intent
        # -------------------------------------------------
        policy_config = self.config_loader.get_policy_for_intent(intent_name)

        # -------------------------------------------------
        # 3️⃣ Evaluate policy (or return default if no policy)
        # -------------------------------------------------
        if not policy_config:
            # No policy defined for this intent - allow normal flow
            policy_output = {
                "business_action": None,
                "required_fields": [],
                "missing_fields": [],
                "allow_execution": True,
                "escalate": False,
                "priority": "normal",
                "reason": f"No policy defined for intent '{intent_name}'"
            }
        else:
            policy_output = self._evaluate_policy(
                policy_config=policy_config,
                intent_name=intent_name,
                sentiment_confidence=sentiment_confidence,
                sentiment_label=sentiment_label,
                context_data=context_data
            )

        # -------------------------------------------------
        # 4️⃣ Return Patch
        # -------------------------------------------------
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,  # Policy evaluation is deterministic
            changes=policy_output
        )

    def _evaluate_policy(
        self,
        policy_config: dict,
        intent_name: str,
        sentiment_confidence: float,
        sentiment_label: str,
        context_data: dict
    ) -> Dict[str, Any]:
        """
        Evaluate a single policy configuration.

        Returns dict with business_action, required_fields, missing_fields,
        allow_execution, escalate, priority, reason.
        """
        # Extract policy settings
        business_action = policy_config.get("business_action")
        required_fields = policy_config.get("required_fields", [])
        blocked_sentiments = policy_config.get("blocked_sentiments", [])
        priority = policy_config.get("priority", "normal")

        # -------------------------------------------------
        # Check 1: Missing required fields
        # -------------------------------------------------
        missing_fields = []
        for field in required_fields:
            if not context_data.get(field):
                missing_fields.append(field)

        # -------------------------------------------------
        # Check 2: Sentiment-based escalation
        # -------------------------------------------------
        escalate = False
        escalate_reason = None

        # Escalate only if sentiment label is in blocked list (e.g., ANGRY)
        # Note: sentiment_confidence is about classification certainty, NOT emotional intensity
        if blocked_sentiments and sentiment_label in blocked_sentiments:
            escalate = True
            escalate_reason = f"Sentiment '{sentiment_label}' is in blocked list"

        # -------------------------------------------------
        # Determine allow_execution
        # -------------------------------------------------
        allow_execution = len(missing_fields) == 0 and not escalate

        # -------------------------------------------------
        # Build reason
        # -------------------------------------------------
        reason_parts = []
        if business_action:
            reason_parts.append(f"Intent '{intent_name}' maps to action '{business_action}'")
        if missing_fields:
            reason_parts.append(f"Missing fields: {missing_fields}")
        if escalate and escalate_reason:
            reason_parts.append(f"Escalate: {escalate_reason}")
        if not escalate and not missing_fields:
            reason_parts.append("All checks passed - execution allowed")

        reason = ". ".join(reason_parts) if reason_parts else f"Policy evaluated for '{intent_name}'"

        # -------------------------------------------------
        # Return policy output
        # -------------------------------------------------
        return {
            "business_action": business_action,
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "allow_execution": allow_execution,
            "escalate": escalate,
            "priority": priority,
            "reason": reason
        }
