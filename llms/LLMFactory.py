"""LLM client factory for creating provider instances."""

from llms.clients.GroqClient import GroqClient
from llms.clients.OpenRouterClient import OpenRouterClient
from llms.LLMManager import LLMManager


class LLMFactory:
    """Factory for creating LLM client instances."""

    _clients = {
        "groq": GroqClient,
        "openrouter": OpenRouterClient,
    }

    @classmethod
    def create(cls, provider: str, **kwargs) -> "LLMManager":
        """Create an LLM client instance.

        Args:
            provider: Provider name (groq, openrouter)
            **kwargs: Additional arguments to pass to client constructor

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is not registered
        """
        provider_lower = provider.lower()

        if provider_lower not in cls._clients:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Available: {list(cls._clients.keys())}"
            )

        client_class = cls._clients[provider_lower]
        return client_class(**kwargs)

    @classmethod
    def register(cls, name: str, client_class: type):
        """Register a new LLM client.

        Args:
            name: Provider name
            client_class: LLM client class
        """
        cls._clients[name.lower()] = client_class