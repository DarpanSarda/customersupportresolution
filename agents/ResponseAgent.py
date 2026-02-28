"""Response agent with async-aware tool result handling."""

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch


class ResponseAgent(BaseAgent):
    """
    Generates final user response based on decision, sentiment, and tool results.

    Handles:
    - ESCALATE: Human handoff message
    - GENERATE_RESPONSE: LLM-generated contextual response
    - Tool results: success, failed, and pending (async)
    """

    agent_name = "ResponseAgent"
    allowed_section = "execution"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """
        Generate final response based on execution state.

        Args:
            state: Current state with decision, execution, and policy sections
            context: Execution context

        Returns:
            Patch with final_response and response_confidence
        """
        # -------------------------------------------------
        # 1️⃣ Check for tool output first (tool-aware)
        # -------------------------------------------------
        execution = state.get("execution", {})
        tools_called = execution.get("tools_called", [])

        if tools_called:
            tool_record = tools_called[-1]
            status = tool_record.get("status")

            # Handle pending (async tool submitted)
            if status == "pending":
                async_job_id = tool_record.get("async_job_id")
                return self._pending_response(async_job_id)

            # Handle tool failure
            if status == "failed":
                error_msg = tool_record.get("error", "Unknown error")
                return self._error_response(error_msg)

            # Handle tool success
            if status == "success":
                return self._success_response(tool_record)

        # -------------------------------------------------
        # 2️⃣ Read decision (primary routing logic)
        # -------------------------------------------------
        decision = state.get("decision")

        # Defensive check: decision section must exist
        if not decision:
            raise ValueError("Decision section missing or empty")

        action = decision.get("action")

        if not action:
            raise ValueError("Decision section missing action")

        # -------------------------------------------------
        # 3️⃣ Handle ESCALATE
        # -------------------------------------------------
        if action == "ESCALATE":
            return self._escalate_response(state)

        # -------------------------------------------------
        # 4️⃣ Handle GENERATE_RESPONSE
        # -------------------------------------------------
        elif action == "GENERATE_RESPONSE":
            # Check if it's a REQUEST_INFO route (missing fields)
            if decision.get("route") == "REQUEST_INFO":
                return self._request_info_response(state)

            # Standard LLM-generated response
            return self._llm_response(state, context)

        else:
            raise ValueError(f"Unknown decision action: {action}")

    def _success_response(self, tool_record: dict) -> Patch:
        """Generate response from successful tool execution."""
        output = tool_record.get("output_payload", {})

        # Handle FAQ tool response
        if output.get("found"):
            final_response = output.get("answer", "Your query was processed successfully.")
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "final_response": final_response,
                    "response_confidence": 1.0
                }
            )

        # Handle generic API response
        if output:
            # Try to find a message field in response
            final_response = (
                output.get("message") or
                output.get("result") or
                output.get("status") or
                "Your request has been processed successfully."
            )

            # Add additional context if available
            if output.get("refund_id"):
                final_response += f" Refund ID: {output['refund_id']}"
            if output.get("ticket_id"):
                final_response += f" Ticket ID: {output['ticket_id']}"
            if output.get("estimated_days"):
                final_response += f" Estimated processing time: {output['estimated_days']} days"

            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "final_response": final_response,
                    "response_confidence": 1.0
                }
            )

        # No useful data in response
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "final_response": "Your request was processed, but no specific information is available.",
                "response_confidence": 0.8
            }
        )

    def _error_response(self, error_msg: str) -> Patch:
        """Generate error response."""
        final_response = (
            f"There was an issue processing your request: {error_msg}. "
            "Please try again later or contact support if the problem persists."
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

    def _pending_response(self, async_job_id: str) -> Patch:
        """Generate response for pending async operation."""
        final_response = (
            f"Your request has been submitted and is being processed (Job ID: {async_job_id}). "
            "You will be notified once it's complete. This usually takes a few minutes."
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

    def _escalate_response(self, state: dict) -> Patch:
        """Generate escalation response."""
        # Check sentiment for tailored escalation message
        sentiment = state.get("understanding", {}).get("sentiment", {})
        sentiment_label = sentiment.get("label", "NEUTRAL")

        if sentiment_label == "ANGRY":
            final_response = (
                "I understand your frustration. Let me connect you to a specialist "
                "who can better assist you with this matter."
            )
        elif sentiment_label == "FRUSTRATED":
            final_response = (
                "I see this is important to you. I'm connecting you to a support "
                "specialist who can help resolve this."
            )
        else:
            final_response = (
                "I'm not confident about handling this request automatically. "
                "Let me connect you to a support specialist."
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

    def _request_info_response(self, state: dict) -> Patch:
        """Generate response requesting missing information."""
        policy = state.get("policy", {})
        missing_fields = policy.get("missing_fields", [])

        # Format field names for user
        readable_fields = {
            "order_id": "order ID",
            "account_id": "account ID",
            "verification_code": "verification code",
            "payment_id": "payment ID",
            "invoice_id": "invoice ID",
            "order_number": "order number",
            "customer_email": "email address"
        }

        readable_missing = [readable_fields.get(f, f) for f in missing_fields]

        final_response = (
            f"To assist you better, I need some additional information. "
            f"Could you please provide: {', '.join(readable_missing)}?"
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

    def _llm_response(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Generate LLM-based contextual response."""
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

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=0.9,
            changes={
                "final_response": final_response,
                "response_confidence": 0.9
            }
        )
