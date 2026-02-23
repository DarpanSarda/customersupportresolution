"""Groq LLM client implementation."""

import os
from groq import Groq
from typing import Dict, List
from models.llm import LLMResponse
from llms.LLMManager import LLMManager
from dotenv import load_dotenv

load_dotenv()

class GroqClient(LLMManager):
    """Groq API client for LLM generation."""

    def __init__(self, model: str = "openai/gpt-oss-120b"):
        """Initialize Groq client.

        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name to use
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = model

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """Generate completion from Groq.

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