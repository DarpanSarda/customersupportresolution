# agents/decision_agent.py

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class DecisionAgent(BaseAgent):
    """Thin router - maps policy output to concrete actions.

    Reads policy section and routes accordingly:
    - policy.escalate = True → ESCALATE
    - policy.allow_execution = True + business_action → CALL_TOOL
    - Otherwise → GENERATE_RESPONSE

    All decision logic is delegated to PolicyAgent.
    DecisionAgent only handles routing based on policy output.
    """

    agent_name = "DecisionAgent"
    allowed_section = "decision"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Route based on policy evaluation.

        Args:
            state: Current state with policy, understanding sections
            context: Execution context (metadata)

        Returns:
            Patch: Decision output with action, route, and reason
        """
        # -------------------------------------------------
        # 1️⃣ Read policy output from PolicyAgent
        # -------------------------------------------------
        policy_data = state.get("policy", {})

        # Fallback: if no policy data, read from understanding (legacy)
        if not policy_data or not policy_data.get("business_action"):
            understanding = state.get("understanding", {})
            intent_data = understanding.get("intent")
            if not intent_data:
                raise ValueError("No policy or understanding data available")

            intent_name = intent_data.get("name")
            confidence = intent_data.get("confidence")

            # Check confidence threshold
            threshold = self.config_loader.get_intent_threshold()
            if confidence < threshold:
                return Patch(
                    agent_name=self.agent_name,
                    target_section=self.allowed_section,
                    confidence=1.0,
                    changes={
                        "action": "ESCALATE",
                        "route": "LOW_CONFIDENCE",
                        "reason": f"Confidence {confidence:.2f} below threshold {threshold}"
                    }
                )

            # No policy defined - check intent tool mapping
            intent_tool_mapping = self.config_loader.get_intent_tool_mapping()
            if intent_name in intent_tool_mapping:
                return Patch(
                    agent_name=self.agent_name,
                    target_section=self.allowed_section,
                    confidence=1.0,
                    changes={
                        "action": "CALL_TOOL",
                        "route": intent_tool_mapping[intent_name],
                        "reason": f"Intent '{intent_name}' mapped to tool"
                    }
                )

            # Default - generate response
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "action": "GENERATE_RESPONSE",
                    "route": "STANDARD",
                    "reason": f"No policy defined for intent '{intent_name}'"
                }
            )

        # -------------------------------------------------
        # 2️⃣ Route based on policy output (new architecture)
        # -------------------------------------------------

        # Priority 1: Escalate if policy says so
        if policy_data.get("escalate"):
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "action": "ESCALATE",
                    "route": "POLICY_ESCALATION",
                    "reason": policy_data.get("reason", "Policy escalation required")
                }
            )

        # Priority 2: Call tool if policy allows execution
        if policy_data.get("allow_execution"):
            business_action = policy_data.get("business_action")

            # Check if business_action maps to a tool
            intent_tool_mapping = self.config_loader.get_intent_tool_mapping()
            if business_action in intent_tool_mapping:
                tool_name = intent_tool_mapping[business_action]
                return Patch(
                    agent_name=self.agent_name,
                    target_section=self.allowed_section,
                    confidence=1.0,
                    changes={
                        "action": "CALL_TOOL",
                        "route": tool_name,
                        "reason": f"Policy allows execution: {business_action} → {tool_name}"
                    }
                )

            # Business action defined but no tool mapping - escalate
            if business_action:
                return Patch(
                    agent_name=self.agent_name,
                    target_section=self.allowed_section,
                    confidence=1.0,
                    changes={
                        "action": "ESCALATE",
                        "route": "NO_TOOL_MAPPING",
                        "reason": f"Business action '{business_action}' has no tool mapping"
                    }
                )

        # Priority 3: Missing fields - generate response asking for them
        missing_fields = policy_data.get("missing_fields", [])
        if missing_fields:
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "action": "GENERATE_RESPONSE",
                    "route": "REQUEST_INFO",
                    "reason": f"Requesting missing fields: {missing_fields}"
                }
            )

        # Priority 4: Default - generate standard response
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "action": "GENERATE_RESPONSE",
                "route": "STANDARD",
                "reason": policy_data.get("reason", "Standard response generation")
            }
        )
