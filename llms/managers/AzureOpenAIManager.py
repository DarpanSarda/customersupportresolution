"""
AzureOpenAIManager - LLM manager for Azure OpenAI Service.

Implements BaseLLM interface for Azure OpenAI's GPT models.
Self-registers with LLMFactory using @register_manager decorator.
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator
from llms.BaseLLM import BaseLLM, LLMConfig, LLMResponse
from llms.LLMFactory import register_manager


@register_manager("azure-openai")
class AzureOpenAIManager(BaseLLM):
    """
    Azure OpenAI LLM manager supporting Azure-deployed GPT models.

    Usage:
        config = LLMConfig(
            provider="azure-openai",
            api_key="...",
            model="gpt-4",
            extra_params={
                "azure_endpoint": "https://your-resource.openai.azure.com/",
                "api_version": "2024-02-15-preview"
            }
        )
        llm = AzureOpenAIManager(config)
        response = await llm.generate("Hello!")
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False

    def _validate_config(self) -> None:
        """Validate Azure OpenAI configuration."""
        if not self.config.api_key:
            raise ValueError("Azure OpenAI API key is required")

        extra = self.config.extra_params or {}
        if not extra.get("azure_endpoint"):
            raise ValueError("azure_endpoint is required in extra_params")

        self._api_version = extra.get("api_version", "2024-02-15-preview")
        self._azure_endpoint = extra["azure_endpoint"]

    async def _initialize_client(self):
        """Lazy initialize the Azure OpenAI client."""
        if self._initialized:
            return

        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI package is required for Azure OpenAI. Install with: pip install openai"
            )

        self._client = openai.AsyncAzureOpenAI(
            api_key=self.config.api_key,
            azure_endpoint=self._azure_endpoint,
            api_version=self._api_version,
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

        extra = self.config.extra_params or {}
        for key in ["azure_endpoint", "api_version"]:
            extra.pop(key, None)
        params.update(extra)
        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider="azure-openai",
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response={
                    "id": response.id,
                    "created": response.created
                }
            )

        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}") from e

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

        extra = self.config.extra_params or {}
        for key in ["azure_endpoint", "api_version"]:
            extra.pop(key, None)
        params.update(extra)
        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)

            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                provider="azure-openai",
                tokens_used=response.usage.total_tokens if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                raw_response={
                    "id": response.id,
                    "created": response.created
                }
            )

        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API error: {str(e)}") from e

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

        extra = self.config.extra_params or {}
        for key in ["azure_endpoint", "api_version"]:
            extra.pop(key, None)
        params.update(extra)
        params.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"Azure OpenAI streaming error: {str(e)}") from e

    async def close(self):
        """Close the Azure OpenAI client."""
        if self._client:
            await self._client.close()
            self._initialized = False
