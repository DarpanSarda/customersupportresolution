"""
BaseLLM - Abstract base class for all LLM managers.

All LLM providers (OpenAI, Claude, Groq, etc.) must implement this interface.
This ensures a unified API regardless of which provider is being used.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict] = None


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: str
    api_key: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    extra_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLM(ABC):
    """
    Abstract base class for LLM managers.

    All LLM provider managers must inherit from this class and implement
    all abstract methods. This ensures a consistent interface across
    different LLM providers.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the provided configuration."""
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from a single prompt string."""
        pass

    @abstractmethod
    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from chat messages."""
        pass

    @abstractmethod
    async def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """Stream a response from a single prompt string."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model being used."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider

    @property
    def client(self):
        """Get the underlying client."""
        return self._client

    async def close(self):
        """Close any open connections or clean up resources."""
        pass

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
