"""
LLMFactory - Dynamic registry-based factory for LLM managers.

Managers self-register using the @register_manager decorator.
The factory creates instances without any hardcoded provider logic.
"""

from typing import Dict, Type
from llms.BaseLLM import BaseLLM, LLMConfig


# Internal registry for all LLM managers
# Managers self-register via the register_manager decorator
_manager_registry: Dict[str, Type[BaseLLM]] = {}


def register_manager(name: str):
    """
    Decorator for LLM managers to self-register.

    Usage in manager class:
        @register_manager("openai")
        class OpenAIManager(BaseLLM):
            ...

    Args:
        name: Provider name (e.g., "openai", "claude", "groq")

    Returns:
        Decorator function that registers the class
    """
    def decorator(cls: Type[BaseLLM]) -> Type[BaseLLM]:
        _manager_registry[name.lower()] = cls
        return cls
    return decorator


def list_registered_managers() -> list:
    """
    Get list of all registered manager names.

    Returns:
        List of registered provider names (lowercase)
    """
    return list(_manager_registry.keys())


def is_manager_registered(provider: str) -> bool:
    """
    Check if a provider is registered.

    Args:
        provider: Provider name to check

    Returns:
        True if provider is registered, False otherwise
    """
    return provider.lower() in _manager_registry


class LLMFactory:
    """
    Dynamic factory for creating LLM manager instances.

    Uses the internal registry to create instances without hardcoding
    any provider logic. New providers can be added by creating a new
    manager class with @register_manager decorator.
    """

    @staticmethod
    def create(provider: str, config: LLMConfig) -> BaseLLM:
        """
        Create an LLM manager instance for the specified provider.

        Args:
            provider: Provider name (e.g., "openai", "claude", "groq")
            config: LLMConfig instance with provider settings

        Returns:
            Instance of the appropriate LLM manager

        Raises:
            ValueError: If provider is not registered
        """
        provider_key = provider.lower()

        if provider_key not in _manager_registry:
            registered = list_registered_managers()
            raise ValueError(
                f"Unknown LLM provider: '{provider}'. "
                f"Registered providers: {registered}"
            )

        manager_class = _manager_registry[provider_key]
        return manager_class(config)

    @staticmethod
    def create_from_dict(config_dict: Dict) -> BaseLLM:
        """
        Create an LLM manager from a configuration dictionary.

        Args:
            config_dict: Dictionary with LLM configuration including:
                - provider: str
                - api_key: str
                - model: str
                - temperature: float (optional)
                - max_tokens: int (optional)
                - ... other LLMConfig fields

        Returns:
            Instance of the appropriate LLM manager

        Raises:
            ValueError: If provider is not registered or required fields missing
        """
        required_fields = ["provider", "api_key", "model"]
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required config field: '{field}'")

        config = LLMConfig(**config_dict)
        return LLMFactory.create(config.provider, config)

    @staticmethod
    def get_registry_info() -> Dict[str, str]:
        """
        Get information about all registered managers.

        Returns:
            Dictionary mapping provider names to their class names
        """
        return {
            name: cls.__name__
            for name, cls in _manager_registry.items()
        }
