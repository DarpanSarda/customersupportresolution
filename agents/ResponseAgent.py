from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class ResponseAgent(BaseAgent):
    """Generates final user response based on decision and sentiment.

    Routes:
    - ESCALATE: Returns human handoff message (tailored based on sentiment)
    - GENERATE_RESPONSE: Uses LLM to generate contextual response with sentiment awareness
    """

    agent_name = "ResponseAgent"
    allowed_section = "execution"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Generate final response based on decision action.

        Args:
            state: Current state with decision section
            context: Execution context

        Returns:
            Patch with final_response and response_confidence
        """
        # -------------------------------------------------
        # 1️⃣ Check for tool output first (failure-aware)
        # -------------------------------------------------
        execution = state.get("execution", {})
        tools_called = execution.get("tools_called", [])

        if tools_called:
            tool_record = tools_called[-1]

            # Handle tool failure
            if tool_record["status"] == "failed":
                final_response = (
                    "There was an issue processing your request. "
                    "Please try again later."
                )
                return Patch(
                    agent_name=self.agent_name,
                    target_section=self.allowed_section,
                    confidence=1.0,
                    changes={
                        "final_response": final_response,
                        "response_confidence": 1.0
                    }
                )

            # Handle tool success
            tool_result = tool_record["output_payload"]
            if tool_result.get("found"):
                final_response = tool_result["answer"]
            else:
                final_response = (
                    "I couldn't find a matching FAQ. "
                    "Could you please provide more details?"
                )

            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "final_response": final_response,
                    "response_confidence": 1.0
                }
            )

        # -------------------------------------------------
        # 2️⃣ Check for policy restrictions (priority override)
        # -------------------------------------------------
        policy = state.get("policy", {})
        restrictions = policy.get("restrictions", [])
        is_compliant = policy.get("compliant", True)

        # If policy has restrictions and is not compliant, use restricted response
        if restrictions and not is_compliant:
            final_response = restrictions[0]  # Use first restriction as response
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "final_response": final_response,
                    "response_confidence": 1.0
                }
            )

        # -------------------------------------------------
        # 3️⃣ Read decision
        # -------------------------------------------------
        decision = state.get("decision")

        # Defensive check: decision section must exist
        if not decision:
            raise ValueError("Decision section missing or empty")

        action = decision.get("action")

        if not action:
            raise ValueError("Decision section missing action")

        # -------------------------------------------------
        # 4️⃣ Handle ESCALATE
        # -------------------------------------------------
        if action == "ESCALATE":
            final_response = (
                "I'm not confident about your request. "
                "Let me connect you to a support specialist."
            )
            response_confidence = 1.0

        # -------------------------------------------------
        # 5️⃣ Handle GENERATE_RESPONSE
        # -------------------------------------------------
        elif action == "GENERATE_RESPONSE":
            understanding = state.get("understanding", {})
            intent = understanding.get("intent", {}).get("name")
            sentiment = understanding.get("sentiment", {})
            user_message = understanding.get("input", {}).get("raw_text")

            # Extract sentiment details (with defaults)
            sentiment_label = sentiment.get("label", "NEUTRAL")
            sentiment_intensity = sentiment.get("intensity", 0.5)

            # Get sentiment response guidelines from config
            sentiment_guidelines = self.config_loader.get_sentiment_response_guidelines()
            sentiment_guideline = sentiment_guidelines.get(sentiment_label, "Be helpful and professional")

            # Load response prompt
            template = self.prompt_loader.load(
                agent_name="response",
                version=self._prompt
            )

            rendered_prompt = self.prompt_loader.render(
                template=template,
                variables={
                    "intent_name": intent,
                    "sentiment_label": sentiment_label,
                    "sentiment_intensity": sentiment_intensity,
                    "sentiment_guideline": sentiment_guideline,
                    "user_message": user_message
                }
            )

            messages = [
                {"role": "system", "content": rendered_prompt}
            ]

            llm_response = self.llm_client.generate(messages)
            final_response = llm_response.content
            response_confidence = 0.9  # placeholder

        else:
            raise ValueError(f"Unknown decision action: {action}")

        # -------------------------------------------------
        # 6️⃣ Return Patch (metadata injected by BaseAgent.execute)
        # -------------------------------------------------
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=response_confidence,
            changes={
                "final_response": final_response,
                "response_confidence": response_confidence
            }
        )