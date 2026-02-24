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
    def create(cls, provider: str | dict, **kwargs) -> "LLMManager":
        """Create an LLM client instance.

        Args:
            provider: Provider name (groq, openrouter) or config dict with 'provider' key
            **kwargs: Additional arguments to pass to client constructor

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider is not registered
        """
        # Handle dict config
        if isinstance(provider, dict):
            provider_name = provider.get("provider")
            if not provider_name:
                raise ValueError("Config dict must contain 'provider' key")
            # Merge dict config with kwargs (kwargs takes precedence)
            config_kwargs = {k: v for k, v in provider.items() if k != "provider"}
            config_kwargs.update(kwargs)
            kwargs = config_kwargs
        else:
            provider_name = provider

        provider_lower = provider_name.lower()

        if provider_lower not in cls._clients:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
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