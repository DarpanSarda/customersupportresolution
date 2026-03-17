"""
ClaudeManager - LLM manager for Anthropic Claude API.

Implements BaseLLM interface for Claude models (Claude 3.5, Claude 3, etc.).
Self-registers with LLMFactory using @register_manager decorator.
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator
from llms.BaseLLM import BaseLLM, LLMConfig, LLMResponse
from llms.LLMFactory import register_manager


@register_manager("claude")
class ClaudeManager(BaseLLM):
    """
    Anthropic Claude LLM manager.

    Usage:
        config = LLMConfig(
            provider="claude",
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022"
        )
        llm = ClaudeManager(config)
        response = await llm.generate("Hello!")
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False

    def _validate_config(self) -> None:
        """Validate Claude configuration."""
        if not self.config.api_key:
            raise ValueError("Claude API key is required")
        if not self.config.api_key.startswith("sk-ant-"):
            raise ValueError("Invalid Claude API key format (should start with 'sk-ant-')")

    async def _initialize_client(self):
        """Lazy initialize the Anthropic client."""
        if self._initialized:
            return

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package is required. Install with: pip install anthropic"
            )

        self._client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            timeout=self.config.timeout
        )
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from a single prompt string."""
        await self._initialize_client()

        params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add any extra params from config or kwargs
        params.update(self.config.extra_params or {})
        params.update(kwargs)

        try:
            response = await self._client.messages.create(**params)

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider="claude",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                raw_response={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            )

        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}") from e

    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from chat messages."""
        await self._initialize_client()

        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        params.update(self.config.extra_params or {})
        params.update(kwargs)

        try:
            response = await self._client.messages.create(**params)

            return LLMResponse(
                content=response.content[0].text,
                model=response.model,
                provider="claude",
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                finish_reason=response.stop_reason,
                raw_response={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    }
                }
            )

        except Exception as e:
            raise RuntimeError(f"Claude API error: {str(e)}") from e

    async def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from a single prompt string."""
        await self._initialize_client()

        params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": True,
        }

        params.update(self.config.extra_params or {})
        params.update(kwargs)

        try:
            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            raise RuntimeError(f"Claude streaming error: {str(e)}") from e

    async def close(self):
        """Close the Anthropic client."""
        if self._client:
            await self._client.close()
            self._initialized = False
