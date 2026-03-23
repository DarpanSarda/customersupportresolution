"""
ResponseAgent - Generates final responses to users.

This agent is responsible for generating the final conversational response
based on all the context gathered by other agents (intent, sentiment, RAG, policy).

According to the Agent Contract:
- MUST only read from shared state
- MUST only return state updates (response, response_type fields)
- MUST NOT call other agents
"""

from typing import Dict, Optional, Any
from datetime import datetime
from core.BaseAgent import BaseAgent
from core.ConversationState import ConversationState, StateUpdate


class ResponseAgent(BaseAgent):
    """
    Response generation agent.

    Reads from shared state and generates the final response.
    Only modifies the 'response' and 'response_type' fields.

    State Dependencies:
        - user_message: The user's input message
        - intent: Detected intent classification
        - sentiment: Detected sentiment
        - rag_context: Retrieved context from knowledge base
        - rag_source_type: Type of RAG source ('faq', 'knowledge_base', None)
        - policy_results: Policy evaluation results

    State Updates:
        - response: The generated response text
        - response_type: Type of response ('faq_answer', 'kb_answer', 'generic', 'escalation')
    """

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry = None
    ):
        """
        Initialize ResponseAgent.

        Args:
            llm_client: LLM client for response generation
            system_prompt: Custom system prompt (optional)
            config: Additional configuration
            tool_registry: Tool registry (optional, not used by this agent)
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful customer support assistant. Your role is to provide clear, accurate, and friendly responses to customer inquiries.

Guidelines:
- Be concise and direct
- Use the provided context when available
- If you don't have specific information, acknowledge it honestly
- Maintain a professional yet approachable tone
- Format responses using markdown when appropriate (tables, lists, etc.)
- For technical terms, provide brief explanations when needed"""

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """
        Process state and generate response.

        Args:
            input_data: ConversationState instance or dict with state fields
            **kwargs: Additional parameters

        Returns:
            StateUpdate with response and response_type fields
        """
        # Handle both ConversationState and dict input
        if isinstance(input_data, ConversationState):
            state = input_data
        else:
            # Legacy dict support - convert to state-like object
            state = self._dict_to_state(input_data)

        # Extract relevant fields from state
        user_message = state.user_message
        intent = state.intent
        sentiment = state.sentiment
        rag_context = state.rag_context
        rag_source_type = state.rag_source_type
        policy_action = state.policy_action

        # Determine response type based on available context
        response_type = self._determine_response_type(rag_source_type, intent, policy_action)

        # Build prompt based on available context
        prompt = self._build_prompt(
            user_message=user_message,
            rag_context=rag_context,
            intent=intent,
            sentiment=sentiment,
            response_type=response_type
        )

        # Generate response
        try:
            llm_response = await self.llm_client.generate(prompt)
            response_text = llm_response.content

        except Exception as e:
            # Fallback to generic response on error
            response_text = self._generate_fallback_response(user_message, intent)
            state.add_warning("ResponseAgent", f"LLM generation failed: {str(e)}")

        # Return state update with only the fields this agent owns
        return StateUpdate(
            response=response_text,
            response_type=response_type
        )

    def _determine_response_type(
        self,
        rag_source_type: Optional[str],
        intent: Optional[str],
        policy_action: Optional[str]
    ) -> str:
        """
        Determine the type of response to generate.

        Args:
            rag_source_type: Source of RAG context
            intent: Detected intent
            policy_action: Recommended policy action

        Returns:
            Response type string
        """
        # If policy recommends escalation
        if policy_action == "escalate":
            return "escalation"

        # If we have FAQ context
        if rag_source_type == "faq":
            return "faq_answer"

        # If we have knowledge base context
        if rag_source_type == "knowledge_base":
            return "kb_answer"

        # No specific context - generic response
        return "generic"

    def _build_prompt(
        self,
        user_message: str,
        rag_context: str,
        intent: Optional[str],
        sentiment: Optional[str],
        response_type: str
    ) -> str:
        """
        Build prompt for LLM based on available context.

        Args:
            user_message: User's input message
            rag_context: Retrieved context from RAG
            intent: Detected intent
            sentiment: Detected sentiment
            response_type: Type of response to generate

        Returns:
            Complete prompt string
        """
        # Start with system prompt (use get_system_prompt() to handle None case)
        prompt_parts = [self.get_system_prompt()]

        # Add context about detected intent/sentiment
        if intent or sentiment:
            context_info = []
            if intent:
                context_info.append(f"Intent: {intent}")
            if sentiment:
                context_info.append(f"Sentiment: {sentiment}")
            # Only add if we have valid context info
            if context_info:
                prompt_parts.append(f"\nDetected: {', '.join(context_info)}")

        # Add RAG context if available
        if rag_context and isinstance(rag_context, str) and rag_context.strip():
            prompt_parts.append(f"""
Context from knowledge base:
{rag_context}

Please answer the user's question based on the context above. If the context doesn't fully address their question, acknowledge what you know and what you don't know.
""")
        else:
            prompt_parts.append("\nNo specific context available from the knowledge base.")

        # Add the user's message
        if user_message:
            prompt_parts.append(f"\nUser: {user_message}")
        else:
            prompt_parts.append("\nUser: [No message provided]")

        # Add response type guidance
        if response_type == "escalation":
            prompt_parts.append("\nNote: This situation may require escalation. Provide a helpful response while acknowledging you'll connect them with specialized support if needed.")
        elif response_type == "generic":
            prompt_parts.append("\nNote: Provide a general helpful response since you don't have specific information about this topic.")

        # Add final instruction
        prompt_parts.append("\nAssistant:")

        # Filter out any None values and ensure all parts are strings
        prompt_parts_filtered = [
            str(part) if part is not None else ""
            for part in prompt_parts
        ]

        return "\n".join(prompt_parts_filtered)

    def _generate_fallback_response(self, user_message: str, intent: Optional[str]) -> str:
        """
        Generate a fallback response when LLM fails.

        Args:
            user_message: User's input message
            intent: Detected intent

        Returns:
            Fallback response string
        """
        if intent == "FAQ_QUERY":
            return "I apologize, but I don't have specific information about that in my knowledge base. For the most accurate and up-to-date information, please check our official documentation or contact our customer support team directly."
        elif intent == "REFUND_REQUEST":
            return "I understand you're asking about a refund. To help you with your refund request, I'll need some additional information about your order. Could you please provide your order number?"
        elif intent == "ORDER_STATUS":
            return "I'd be happy to help you check your order status. Please provide your order number so I can look that up for you."
        else:
            return f"I understand you're asking about: {user_message}. Let me help you with that. How can I assist you further?"

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
            user_message=data.get("message", data.get("user_message", "")),
            intent=data.get("intent"),
            sentiment=data.get("sentiment"),
            rag_context=data.get("rag_context", ""),
            rag_source_type=data.get("rag_source_type"),
            policy_action=data.get("policy_action"),
            add_error=lambda *args: None  # No-op for dict input
        )

    @classmethod
    def get_agent_info(cls) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "ResponseAgent",
            "description": "Generates final conversational responses based on gathered context",
            "input_fields": [
                "user_message", "intent", "sentiment",
                "rag_context", "rag_source_type", "policy_action"
            ],
            "output_fields": ["response", "response_type"],
            "state_modifications": ["response", "response_type"]
        }
