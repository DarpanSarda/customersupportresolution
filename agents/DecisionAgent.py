# agents/decision_agent.py

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class DecisionAgent(BaseAgent):
    """Deterministic decision-making agent.

    Applies confidence-based routing rules:
    - Low confidence: ESCALATE to human
    - High confidence: GENERATE_RESPONSE via Resolution Agent

    Config-driven thresholds via config_loader.
    """

    agent_name = "DecisionAgent"
    allowed_section = "decision"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        # Prompt not used for rule-based agent, but stored for consistency

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Apply decision rules based on intent confidence.

        Args:
            state: Current state with understanding section
            context: Execution context (metadata)

        Returns:
            Patch: Decision output with action, route, and reason
        """
        # -------------------------------------------------
        # 1️⃣ Read understanding
        # -------------------------------------------------

        understanding = state.get("understanding", {})
        intent_data = understanding.get("intent")

        if not intent_data:
            raise ValueError("Understanding section missing intent")

        intent_name = intent_data.get("name")
        confidence = intent_data.get("confidence")

        # Defensive check: confidence must exist
        if confidence is None:
            raise ValueError("Confidence missing in understanding section")

        # -------------------------------------------------
        # 2️⃣ Get threshold from config
        # -------------------------------------------------

        threshold = self.config_loader.get_intent_threshold()

        # -------------------------------------------------
        # 3️⃣ Apply routing rules
        # -------------------------------------------------

        # Check for FAQ intent first - route to tool
        if intent_name == "FAQ_QUERY":
            decision = {
                "action": "CALL_TOOL",
                "route": "faq_lookup",
                "reason": "FAQ intent detected - using lookup tool"
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
        # 4️⃣ Return Patch (metadata injected by BaseAgent.execute)
        # -------------------------------------------------

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,  # deterministic rule-based agent
            changes=decision
            # metadata will be injected by BaseAgent.execute()
        )