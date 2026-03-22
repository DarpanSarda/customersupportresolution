"""
SentimentAgent - Analyzes user messages for emotional tone and urgency.

Uses LLM with structured prompts to detect sentiment, urgency, and toxicity.
Returns a ResponsePatch with sentiment data.
"""

import json
import re
from datetime import datetime, timezone
from typing import Dict, Any
from core.BaseAgent import BaseAgent
from schemas.response import ResponsePatch


class SentimentAgent(BaseAgent):
    """
    Sentiment analysis agent.

    Analyzes user messages to detect emotional tone, urgency levels,
    and potential toxicity. Used for escalation decisions and priority routing.

    Returns a ResponsePatch with patch_type="sentiment".
    """

    def _get_default_system_prompt(self) -> str:
        """Get default sentiment analysis prompt."""
        raise ValueError("Set the sentiment prompt before continuing")

    async def process(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> ResponsePatch:
        """
        Process sentiment analysis request.

        This is the main BaseAgent interface method.

        Args:
            input_data: Dictionary containing:
                - message: User's message to analyze
                - conversation_history: Optional recent conversation context
            **kwargs: Additional parameters for LLM

        Returns:
            ResponsePatch with sentiment analysis data
        """
        # Check if prompt is configured
        try:
            self.get_system_prompt()
        except ValueError as e:
            return ResponsePatch(
                agent_name=self.get_agent_name(),
                patch_type="sentiment",
                data={
                    "sentiment": "neutral",
                    "urgency_score": 0.0,
                    "toxicity_flag": False,
                    "error": str(e)
                },
                confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        message = input_data.get("message")
        conversation_history = input_data.get("conversation_history", [])

        if not message:
            return ResponsePatch(
                agent_name=self.get_agent_name(),
                patch_type="sentiment",
                data={
                    "sentiment": "neutral",
                    "urgency_score": 0.0,
                    "toxicity_flag": False,
                    "error": "No message provided"
                },
                confidence=0.0,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

        result = await self._analyze(message, conversation_history, **kwargs)

        # Return ResponsePatch
        return ResponsePatch(
            agent_name=self.get_agent_name(),
            patch_type="sentiment",
            data={
                "sentiment": result["sentiment"],
                "urgency_score": result["urgency_score"],
                "toxicity_flag": result.get("toxicity_flag", False),
                "reasoning": result.get("reasoning", ""),
                "emotional_indicators": result.get("emotional_indicators", [])
            },
            confidence=result.get("confidence", 0.8),
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    async def _analyze(
        self,
        message: str,
        conversation_history: list = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze message for sentiment and urgency.

        Args:
            message: User's message to analyze
            conversation_history: Recent conversation context
            **kwargs: Additional parameters for LLM

        Returns:
            Dictionary with sentiment analysis details
        """
        # Build context from conversation history
        context = self._build_context(message, conversation_history)

        # Build the prompt
        user_prompt = self._build_analysis_prompt(context)

        # Get LLM response
        response = await self.llm_client.generate_with_messages(
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Low temperature for consistent analysis
            max_tokens=200,
            **kwargs
        )

        # Parse the response
        return self._parse_sentiment_response(response.content)

    def _build_context(self, message: str, conversation_history: list = None) -> str:
        """
        Build context string from message and history.

        Args:
            message: Current message
            conversation_history: Recent conversation history

        Returns:
            Context string for analysis
        """
        if not conversation_history:
            return message

        # Get last 3 messages for context
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history

        context_parts = []
        for msg in recent_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            context_parts.append(f"{role.capitalize()}: {content}")

        context_parts.append(f"Current User Message: {message}")
        return "\n".join(context_parts)

    def _build_analysis_prompt(self, context: str) -> str:
        """
        Build the user prompt for sentiment analysis.

        Args:
            context: Message context to analyze

        Returns:
            Prompt string
        """
        return f"""Analyze the following message(s) for sentiment and urgency:

{context}

Provide your analysis in JSON format."""

    def _parse_sentiment_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse LLM response into sentiment data.

        Args:
            response_content: Raw LLM response

        Returns:
            Dictionary with sentiment analysis data
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_dict = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")

            sentiment = response_dict.get("sentiment", "neutral").lower()
            urgency = float(response_dict.get("urgency_score", 0.0))
            toxicity = response_dict.get("toxicity_flag", False)

            # Normalize sentiment to valid values
            valid_sentiments = ["angry", "frustrated", "neutral", "positive"]
            if sentiment not in valid_sentiments:
                sentiment = "neutral"

            # Calculate urgency score based on sentiment
            urgency_score = self._calculate_urgency(sentiment, urgency)

            return {
                "sentiment": sentiment,
                "urgency_score": urgency_score,
                "toxicity_flag": toxicity or sentiment == "angry",
                "confidence": response_dict.get("confidence", 0.8),
                "reasoning": response_dict.get("reasoning", ""),
                "emotional_indicators": response_dict.get("emotional_indicators", [])
            }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback to neutral on parsing error
            return {
                "sentiment": "neutral",
                "urgency_score": 0.0,
                "toxicity_flag": False,
                "confidence": 0.3,
                "reasoning": f"Parse error: {str(e)}",
                "emotional_indicators": []
            }

    def _calculate_urgency(self, sentiment: str, detected_urgency: float) -> float:
        """
        Calculate urgency score based on sentiment and detected urgency.

        Args:
            sentiment: Detected sentiment
            detected_urgency: Urgency score from LLM

        Returns:
            Normalized urgency score (0-1)
        """
        # Base urgency by sentiment
        base_urgency = {
            "angry": 0.9,
            "frustrated": 0.7,
            "neutral": 0.3,
            "positive": 0.1
        }.get(sentiment, 0.3)

        # Combine with detected urgency (average)
        return (base_urgency + detected_urgency) / 2
