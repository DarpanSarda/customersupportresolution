# agents/decision_agent.py

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class DecisionAgent(BaseAgent):
    """Deterministic decision-making agent.

    Applies decision rules in priority order:
    1. Policy-based escalation (violations → ESCALATE)
    2. Sentiment-based escalation (ANGRY/FRUSTRATED → ESCALATE)
    3. Intent-based routing (tool mapping, confidence threshold)

    All thresholds and policies are config-driven.
    """

    agent_name = "DecisionAgent"
    allowed_section = "decision"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        # Prompt not used for rule-based agent, but stored for consistency

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Apply decision rules based on policy, sentiment, and intent.

        Args:
            state: Current state with policy, understanding sections
            context: Execution context (metadata)

        Returns:
            Patch: Decision output with action, route, and reason
        """
        # -------------------------------------------------
        # 1️⃣ Read policy and understanding
        # -------------------------------------------------

        policy_data = state.get("policy", {})
        understanding = state.get("understanding", {})
        intent_data = understanding.get("intent")
        sentiment_data = understanding.get("sentiment")

        if not intent_data:
            raise ValueError("Understanding section missing intent")

        intent_name = intent_data.get("name")
        confidence = intent_data.get("confidence")

        # Defensive check: confidence must exist
        if confidence is None:
            raise ValueError("Confidence missing in understanding section")

        # -------------------------------------------------
        # 2️⃣ Get thresholds from config
        # -------------------------------------------------

        threshold = self.config_loader.get_intent_threshold()
        escalation_thresholds = self.config_loader.get_sentiment_escalation_thresholds()

        # -------------------------------------------------
        # 3️⃣ Apply policy-based escalation (HIGHEST PRIORITY)
        # -------------------------------------------------

        if policy_data:
            is_compliant = policy_data.get("compliant", True)
            violations = policy_data.get("violations", [])

            if not is_compliant and violations:
                # Check severity for escalation decision
                has_critical = any(v.get("severity") == "CRITICAL" for v in violations)
                has_high = any(v.get("severity") == "HIGH" for v in violations)

                if has_critical or has_high:
                    violation_names = [v.get("policy_name") for v in violations]
                    return Patch(
                        agent_name=self.agent_name,
                        target_section=self.allowed_section,
                        confidence=1.0,
                        changes={
                            "action": "ESCALATE",
                            "route": "POLICY_VIOLATION",
                            "reason": f"Policy violation detected: {', '.join(violation_names)}"
                        }
                    )

        # -------------------------------------------------
        # 4️⃣ Apply sentiment-based escalation
        # -------------------------------------------------

        if sentiment_data and escalation_thresholds:
            sentiment_label = sentiment_data.get("label")
            sentiment_confidence = sentiment_data.get("confidence")

            if sentiment_label in escalation_thresholds:
                threshold_value = escalation_thresholds[sentiment_label]
                if sentiment_confidence >= threshold_value:
                    return Patch(
                        agent_name=self.agent_name,
                        target_section=self.allowed_section,
                        confidence=1.0,
                        changes={
                            "action": "ESCALATE",
                            "route": "HIGH_NEGATIVE_SENTIMENT",
                            "reason": (
                                f"High {sentiment_label} sentiment detected "
                                f"({sentiment_confidence:.2f} >= {threshold_value})"
                            )
                        }
                    )

        # -------------------------------------------------
        # 5️⃣ Apply intent-based routing rules
        # -------------------------------------------------

        # Check if intent maps to a tool (config-driven)
        intent_tool_mapping = self.config_loader.get_intent_tool_mapping()
        if intent_name in intent_tool_mapping:
            tool_name = intent_tool_mapping[intent_name]
            decision = {
                "action": "CALL_TOOL",
                "route": tool_name,
                "reason": f"Intent '{intent_name}' mapped to tool '{tool_name}'"
            }
        elif confidence < threshold:
            decision = {
                "action": "ESCALATE",
                "route": "LOW_CONFIDENCE",
                "reason": f"Confidence {confidence:.2f} below threshold {threshold}"
            }
        else:
            decision = {
                "action": "GENERATE_RESPONSE",
                "route": "STANDARD",
                "reason": f"Valid intent '{intent_name}' detected with confidence {confidence:.2f}"
            }

        # -------------------------------------------------
        # 6️⃣ Return Patch (metadata injected by BaseAgent.execute)
        # -------------------------------------------------

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,  # deterministic rule-based agent
            changes=decision
            # metadata will be injected by BaseAgent.execute()
        )