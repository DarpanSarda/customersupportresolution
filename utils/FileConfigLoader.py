"""
FileConfigLoader - Load configurations from file system.

Configuration file structure:
    config/
    ├── intents.json
    ├── sentiments.json
    ├── policies.json
    └── ...
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional


class FileConfigLoader:
    """
    Load configurations from file system.

    Supports JSON configuration files for intents, sentiments, policies, etc.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize FileConfigLoader.

        Args:
            config_dir: Directory containing configuration files (default: "config")
        """
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}

    def _read_json_file(self, file_path: Path) -> Any:
        """
        Read JSON file and return parsed data.

        Args:
            file_path: Path to JSON file

        Returns:
            Parsed JSON data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed as JSON
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file {file_path}: {str(e)}")

    def get_intents(self, tenant_id: str = "default") -> List[Dict[str, Any]]:
        """
        Get intent labels for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of intent configurations

        Raises:
            FileNotFoundError: If intents configuration file not found
        """
        cache_key = f"intents:{tenant_id}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try tenant-specific file first
        tenant_file = self.config_dir / f"intents_{tenant_id}.json"
        default_file = self.config_dir / "intents.json"

        file_path = tenant_file if tenant_file.exists() else default_file

        if not file_path.exists():
            raise FileNotFoundError(
                f"Intents configuration file not found: {default_file}\n"
                f"Please create the config directory and add an intents.json file."
            )

        data = self._read_json_file(file_path)

        # Handle both list and dict formats
        if isinstance(data, dict):
            intents_data = data.get("intents", data.get("intent_labels", []))
        else:
            intents_data = data

        if not intents_data:
            raise ValueError(f"No intents found in configuration file: {file_path}")

        self._cache[cache_key] = intents_data
        return intents_data

    def get_sentiment_config(self, tenant_id: str = "default") -> Dict[str, Any]:
        """
        Get sentiment configuration for a tenant.

        Returns the full sentiment config including sentiments list and escalation rules.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Dictionary with:
                - sentiments: List of sentiment configurations
                - escalation_rules: Escalation rules dictionary
        """
        cache_key = f"sentiments:{tenant_id}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try tenant-specific file first
        tenant_file = self.config_dir / f"sentiments_{tenant_id}.json"
        default_file = self.config_dir / "sentiments.json"

        file_path = tenant_file if tenant_file.exists() else default_file

        if not file_path.exists():
            # Return default sentiment config if file doesn't exist
            default_sentiments = self._get_default_sentiments()
            config = {
                "sentiments": default_sentiments,
                "escalation_rules": {}
            }
            self._cache[cache_key] = config
            return config

        data = self._read_json_file(file_path)

        # Handle both list and dict formats
        if isinstance(data, list):
            # Simple list format - no escalation rules
            config = {
                "sentiments": data,
                "escalation_rules": {}
            }
        else:
            # Full config format with escalation rules
            config = {
                "sentiments": data.get("sentiments", data.get("sentiment_configs", [])),
                "escalation_rules": data.get("escalation_rules", {})
            }

        self._cache[cache_key] = config
        return config

    def get_policy(self, tenant_id: str, intent_type: str) -> Optional[Dict[str, Any]]:
        """
        Get policy for tenant and intent.

        Args:
            tenant_id: Tenant identifier
            intent_type: Intent type

        Returns:
            Policy configuration or None if not found
        """
        cache_key = f"policies:{tenant_id}"

        if cache_key not in self._cache:
            # Load all policies for tenant
            tenant_file = self.config_dir / f"policies_{tenant_id}.json"
            default_file = self.config_dir / "policies.json"

            file_path = tenant_file if tenant_file.exists() else default_file

            if file_path.exists():
                data = self._read_json_file(file_path)

                # Handle both list and dict formats
                if isinstance(data, dict):
                    policies_data = data.get("policies", [])
                else:
                    policies_data = data

                # Index by intent_type for quick lookup
                self._cache[cache_key] = {
                    p.get("intent_type", p.get("intent", "")): p
                    for p in policies_data
                }
            else:
                self._cache[cache_key] = {}

        policies = self._cache[cache_key]
        return policies.get(intent_type)

    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tenant configuration.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant configuration or None if not found
        """
        file_path = self.config_dir / "tenants.json"

        if not file_path.exists():
            return None

        data = self._read_json_file(file_path)

        # Handle both list and dict formats
        if isinstance(data, dict):
            tenants_data = data.get("tenants", [])
        else:
            tenants_data = data

        # Find tenant by ID
        for tenant in tenants_data:
            if tenant.get("tenant_id") == tenant_id:
                return tenant

        return None

    def _get_default_sentiments(self) -> List[Dict[str, Any]]:
        """
        Get default sentiment configuration.

        Returns:
            Default sentiment configurations
        """
        return [
            {
                "sentiment_label": "positive",
                "escalation_threshold": None,
                "response_guideline": "Acknowledge and thank the customer"
            },
            {
                "sentiment_label": "neutral",
                "escalation_threshold": None,
                "response_guideline": "Provide helpful assistance"
            },
            {
                "sentiment_label": "negative",
                "escalation_threshold": 0.8,
                "response_guideline": "Be empathetic and consider escalation"
            },
            {
                "sentiment_label": "angry",
                "escalation_threshold": 0.6,
                "response_guideline": "Immediately escalate to human agent"
            }
        ]

    def clear_cache(self, key: Optional[str] = None) -> None:
        """
        Clear configuration cache.

        Args:
            key: Specific cache key to clear, or None to clear all
        """
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


# Global instance
_config_loader: Optional[FileConfigLoader] = None


def get_config_loader(config_dir: str = "config") -> FileConfigLoader:
    """
    Get the global FileConfigLoader instance.

    Args:
        config_dir: Directory containing configuration files

    Returns:
        FileConfigLoader instance
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = FileConfigLoader(config_dir)

    return _config_loader
