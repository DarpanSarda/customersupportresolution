"""OpenRouter LLM client implementation."""

import os
from openai import OpenAI
from typing import Dict, List
from models.llm import LLMResponse
from llms.LLMManager import LLMManager
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient(LLMManager):
    """OpenRouter API client for LLM generation."""

    def __init__(self, model: str = "anthropic/claude-3.5-sonnet"):
        """Initialize OpenRouter client.

        Args:
            model: Model name to use
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion from OpenRouter.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with content, model, usage, and raw response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump() if response.usage else None,
            raw=response.model_dump() if hasattr(response, 'model_dump') else None
        )