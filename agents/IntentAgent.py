"""
IntentAgent - Classifies user messages into predefined intents.

Uses LLM with structured prompts to classify user intent and confidence.
Returns a ResponsePatch with intent data.
"""

import json
import re
from datetime import datetime, timezone
from typing import List, Dict, Any
from core.BaseAgent import BaseAgent
from schemas.intent import IntentLabel
from schemas.response import ResponsePatch


class IntentAgent(BaseAgent):
    """
    Intent classification agent.

    Analyzes user messages and classifies them into predefined intents
    like GREETING, FAQ_QUERY, REFUND_REQUEST, ORDER_STATUS, etc.

    Returns a ResponsePatch with patch_type="intent".
    """

    def _get_default_system_prompt(self) -> str:
        """Get default intent classification prompt."""
        raise ValueError("Set the intent prompt before continuing")

    async def process(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> ResponsePatch:
        """
        Process intent classification request.

        This is the main BaseAgent interface method.

        Args:
            input_data: Dictionary containing:
                - message: User's message
                - available_intents: List of available intents
            **kwargs: Additional parameters for LLM

        Returns:
            ResponsePatch with intent classification data
        """
        # Check if prompt is configured
        try:
            self.get_system_prompt()
        except ValueError as e:
            return ResponsePatch(
                agent_name=self.get_agent_name(),
                patch_type="intent",
                data={
                    "intent": None,
                    "confidence": 0.0,
                    "meets_threshold": False,
                    "error": str(e)
                },
                confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        message = input_data.get("message")
        available_intents = input_data.get("available_intents", [])

        if not message:
            return ResponsePatch(
                agent_name=self.get_agent_name(),
                patch_type="intent",
                data={
                    "intent": "GENERAL_QUERY",
                    "confidence": 0.0,
                    "meets_threshold": False,
                    "error": "No message provided"
                },
                confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        result = await self._classify(message, available_intents, **kwargs)

        # Return ResponsePatch
        return ResponsePatch(
            agent_name=self.get_agent_name(),
            patch_type="intent",
            data={
                "intent": result["intent"],
                "confidence": result["confidence"],
                "meets_threshold": result["meets_threshold"],
                "tool_mapping": result.get("tool_mapping"),
                "reasoning": result.get("reasoning", "")
            },
            confidence=result["confidence"],
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    async def _classify(
        self,
        message: str,
        available_intents: List[IntentLabel],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Classify user message into an intent.

        Args:
            message: User's message
            available_intents: List of available intents to classify into
            **kwargs: Additional parameters for LLM

        Returns:
            Dictionary with classification details
        """
        # Build intent descriptions
        intent_descriptions = self._build_intent_descriptions(available_intents)

        # Build the prompt
        user_prompt = self._build_classification_prompt(message, intent_descriptions)

        # Get LLM response
        response = await self.llm_client.generate_with_messages(
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent classification
            max_tokens=200,
            **kwargs
        )

        # Parse the response
        return self._parse_intent_response(
            response.content,
            available_intents
        )

    def _build_intent_descriptions(self, intents: List[IntentLabel]) -> str:
        """Build formatted intent descriptions."""
        descriptions = []
        for intent in intents:
            desc = f"- {intent.label}"
            if intent.description:
                desc += f": {intent.description}"
            if intent.examples:
                desc += f" (e.g., {', '.join(intent.examples[:3])})"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def _build_classification_prompt(self, message: str, intent_descriptions: str) -> str:
        """Build the user prompt for classification."""
        return f"""Classify the following user message into ONE of these intents:

{intent_descriptions}

User Message: "{message}"

Respond in JSON format with intent, confidence, and reasoning."""

    def _parse_intent_response(
        self,
        response_content: str,
        available_intents: List[IntentLabel]
    ) -> Dict[str, Any]:
        """
        Parse LLM response into intent data.

        Args:
            response_content: Raw LLM response
            available_intents: List of available intents for validation

        Returns:
            Dictionary with intent data
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_dict = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            intent = response_dict.get("intent", "GENERAL_QUERY")
            confidence = float(response_dict.get("confidence", 0.5))

            # Validate intent exists
            intent_labels = [i.label for i in available_intents]
            if intent not in intent_labels:
                intent = "GENERAL_QUERY"
                confidence = 0.3

            # Find the intent label to get threshold and tool mapping
            intent_obj = next((i for i in available_intents if i.label == intent), None)
            threshold = intent_obj.confidence_threshold if intent_obj else 0.7
            tool_mapping = intent_obj.tool_mapping if intent_obj else None

            # Check if meets threshold
            meets_threshold = confidence >= threshold

            return {
                "intent": intent,
                "confidence": confidence,
                "meets_threshold": meets_threshold,
                "tool_mapping": tool_mapping,
                "reasoning": response_dict.get("reasoning", "")
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to general query on parsing error
            return {
                "intent": "GENERAL_QUERY",
                "confidence": 0.3,
                "meets_threshold": False,
                "tool_mapping": None,
                "reasoning": f"Parse error: {str(e)}"
            }
