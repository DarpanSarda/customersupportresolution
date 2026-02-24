from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class FallbackAgent(BaseAgent):
    """Deterministic fallback agent for system failures.

    Used when tool execution fails or system encounters errors.
    Returns a generic fallback message without calling LLM.
    """

    agent_name = "FallbackAgent"
    allowed_section = "execution"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        # Prompt not used for deterministic fallback

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Return deterministic fallback message.

        Args:
            state: Current state (not used, kept for interface consistency)
            context: Execution context (not used, kept for interface consistency)

        Returns:
            Patch with fallback message
        """
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "final_response": "We encountered a system issue. Please try again later.",
                "response_confidence": 1.0
            }
        )