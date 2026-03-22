"""
OpenAIManager - Manager for OpenAI API only.

This manager is specifically for the official OpenAI API with strict validation.
For OpenAI-compatible APIs (Groq, OpenRouter, etc.), use OpenAICompatibleManager.
"""

from typing import Dict, List, Optional, Any, AsyncIterator
from llms.BaseLLM import BaseLLM, LLMConfig, LLMResponse
from llms.LLMFactory import register_manager


@register_manager("openai")
class OpenAIManager(BaseLLM):
    """
    Manager for OpenAI API only.

    Usage:
        config = LLMConfig(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o"
        )

        llm = OpenAIManager(config)
        response = await llm.generate("Hello!")
    """

    def _validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")

        # Strict validation for OpenAI API keys
        if not self.config.api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format (should start with 'sk-')")

    async def _initialize_client(self):
        """Lazy initialize the OpenAI client."""
        if self._initialized:
            return

        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self._client = openai.AsyncOpenAI(
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
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }

        # Add extra params if provided
        if self.config.extra_params:
            extra = {k: v for k, v in self.config.extra_params.items()
                    if k not in ["base_url", "azure_endpoint", "api_version"]}
            params.update(extra)

        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)

            raw_response = {
                "id": response.id,
                "created": response.created
            }

            if hasattr(response, "system_fingerprint") and response.system_fingerprint:
                raw_response["system_fingerprint"] = response.system_fingerprint

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=self.config.provider,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=raw_response
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

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
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
        }

        if self.config.extra_params:
            extra = {k: v for k, v in self.config.extra_params.items()
                    if k not in ["base_url", "azure_endpoint", "api_version"]}
            params.update(extra)

        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)

            raw_response = {
                "id": response.id,
                "created": response.created
            }

            if hasattr(response, "system_fingerprint") and response.system_fingerprint:
                raw_response["system_fingerprint"] = response.system_fingerprint

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider=self.config.provider,
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response=raw_response
            )

        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e

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
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": True,
        }

        if self.config.extra_params:
            extra = {k: v for k, v in self.config.extra_params.items()
                    if k not in ["base_url", "azure_endpoint", "api_version"]}
            params.update(extra)

        params.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"OpenAI streaming error: {str(e)}") from e

    async def close(self):
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._initialized = False
