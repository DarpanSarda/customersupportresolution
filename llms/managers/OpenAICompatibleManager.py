"""
OpenAICompatibleManager - Unified manager for OpenAI and OpenAI-compatible APIs.

Handles:
- OpenAI (default base URL)
- Azure OpenAI (with azure_endpoint)
- Groq, OpenRouter, DeepSeek, Qwen, Together, Anyscale, Perplexity, Mistral
- Any OpenAI-compatible API with custom base_url

All providers use the same OpenAI SDK with optional base_url/azure_endpoint.
Self-registers multiple provider aliases with LLMFactory.
"""

from typing import Dict, List, Optional, Any, AsyncIterator
from llms.BaseLLM import BaseLLM, LLMConfig, LLMResponse

# Base URLs for popular OpenAI-compatible providers
PROVIDER_BASE_URLS = {
    "openai": None,  # OpenAI's default (no base_url needed)
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "together": "https://api.together.xyz/v1",
    "anyscale": "https://api.endpoints.anyscale.com/v1",
    "perplexity": "https://api.perplexity.ai",
    "mistral": "https://api.mistral.ai/v1",
}

class OpenAICompatibleManager(BaseLLM):
    """
    Unified manager for OpenAI and all OpenAI-compatible API providers.

    Usage:
        # OpenAI (default)
        config = LLMConfig(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o"
        )

        # Groq
        config = LLMConfig(
            provider="groq",
            api_key="gsk_...",
            model="llama-3.3-70b-versatile"
        )

        # OpenRouter
        config = LLMConfig(
            provider="openrouter",
            api_key="sk-or-...",
            model="anthropic/claude-3.5-sonnet"
        )

        # Azure OpenAI
        config = LLMConfig(
            provider="azure-openai",
            api_key="...",
            model="gpt-4",
            extra_params={
                "azure_endpoint": "https://your-resource.openai.azure.com/",
                "api_version": "2024-02-15-preview"
            }
        )

        # Custom base URL
        config = LLMConfig(
            provider="custom",
            api_key="...",
            model="...",
            extra_params={"base_url": "https://your-api.com/v1"}
        )

        llm = OpenAICompatibleManager(config)
        response = await llm.generate("Hello!")
    """

    _registrations_done = False

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False

        # Determine base URL and azure endpoint
        self._base_url = None
        self._azure_endpoint = None
        self._api_version = None

        if config.extra_params:
            if "base_url" in config.extra_params:
                self._base_url = config.extra_params["base_url"]
            elif "azure_endpoint" in config.extra_params:
                self._azure_endpoint = config.extra_params["azure_endpoint"]
                self._api_version = config.extra_params.get("api_version", "2024-02-15-preview")
            elif config.provider.lower() in PROVIDER_BASE_URLS:
                self._base_url = PROVIDER_BASE_URLS[config.provider.lower()]
        elif config.provider.lower() in PROVIDER_BASE_URLS:
            self._base_url = PROVIDER_BASE_URLS[config.provider.lower()]

    @classmethod
    def register_all_providers(cls):
        """Register all provider aliases for this manager."""
        if cls._registrations_done:
            return

        # Register each provider name
        all_providers = list(PROVIDER_BASE_URLS.keys()) + ["azure-openai"]

        for provider in all_providers:
            # Create a derived class for each provider
            derived_class = type(
                f"{provider.capitalize().replace('-', '')}Manager",
                (cls,),
                {"__module__": cls.__module__}
            )
            # Register with the factory's internal registry
            from llms.LLMFactory import _manager_registry
            _manager_registry[provider] = derived_class

        cls._registrations_done = True

    def _validate_config(self) -> None:
        """Validate configuration."""
        if not self.config.api_key:
            raise ValueError(f"{self.config.provider} API key is required")

        # Validate OpenAI key format for openai provider
        if self.config.provider == "openai" and not self.config.api_key.startswith("sk-"):
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

        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout
        }

        # Add base_url if we have one
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        # Create the appropriate client
        if self._azure_endpoint:
            self._client = openai.AsyncAzureOpenAI(
                api_key=self.config.api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
                timeout=self.config.timeout
            )
        else:
            self._client = openai.AsyncOpenAI(**client_kwargs)

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

        # Clean extra_params and add to params
        extra = self.config.extra_params or {}
        extra = {k: v for k, v in extra.items() if k not in ["base_url", "azure_endpoint", "api_version"]}
        params.update(extra)
        params.update(kwargs)

        try:
            response = await self._client.chat.completions.create(**params)

            raw_response = {
                "id": response.id,
                "created": response.created
            }

            # Add system_fingerprint if available (OpenAI specific)
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
            raise RuntimeError(f"{self.config.provider} API error: {str(e)}") from e

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
        extra = {k: v for k, v in extra.items() if k not in ["base_url", "azure_endpoint", "api_version"]}
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
            raise RuntimeError(f"{self.config.provider} API error: {str(e)}") from e

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
        extra = {k: v for k, v in extra.items() if k not in ["base_url", "azure_endpoint", "api_version"]}
        params.update(extra)
        params.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise RuntimeError(f"{self.config.provider} streaming error: {str(e)}") from e

    async def close(self):
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._initialized = False


# Auto-register all providers on import
OpenAICompatibleManager.register_all_providers()
