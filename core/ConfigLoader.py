# core/config_loader.py

class ConfigLoader:

    def __init__(self, config: dict):
        self._config = config

    # -----------------------------------------------------
    # Generic Access
    # -----------------------------------------------------

    def get(self, key: str, default=None):
        return self._config.get(key, default)

    # -----------------------------------------------------
    # LLM Config
    # -----------------------------------------------------

    def get_llm_config(self):
        return self._config.get("llm", {})

    # -----------------------------------------------------
    # Agent Config
    # -----------------------------------------------------

    def get_agent_config(self, agent_name: str):
        return self._config.get("agents", {}).get(agent_name, {})

    # -----------------------------------------------------
    # Intent Config
    # -----------------------------------------------------

    def get_intent_labels(self):
        return self._config.get("intent", {}).get("labels", [])

    def get_intent_threshold(self):
        return self._config.get("intent", {}).get(
            "confidence_threshold", 0.0
        )

    def get_intent_tool_mapping(self):
        """Get intent to tool mapping for routing."""
        return self._config.get("intent", {}).get("tool_mapping", {})

    # -----------------------------------------------------
    # Sentiment Config
    # -----------------------------------------------------

    def get_sentiment_labels(self):
        """Get available sentiment classification labels."""
        return self._config.get("sentiment", {}).get("labels", [])

    def get_sentiment_escalation_thresholds(self):
        """Get sentiment thresholds for auto-escalation."""
        return self._config.get("sentiment", {}).get("escalation_thresholds", {})

    def get_sentiment_response_guidelines(self):
        """Get sentiment response guidelines for tailoring responses."""
        return self._config.get("sentiment", {}).get("response_guidelines", {})

    # -----------------------------------------------------
    # Policy Config
    # -----------------------------------------------------

    def is_policy_enabled(self):
        """Check if policy evaluation is enabled."""
        return self._config.get("policy", {}).get("enabled", True)

    def get_policies(self):
        """Get all configured policies."""
        return self._config.get("policy", {}).get("policies", {})

    def get_policy(self, policy_name: str):
        """Get specific policy by name."""
        return self.get_policies().get(policy_name, {})

    def get_severity_levels(self):
        """Get severity level definitions."""
        return self._config.get("policy", {}).get("severity_levels", {})

    # -----------------------------------------------------
    # Routing Config
    # -----------------------------------------------------

    def get_routing_config(self):
        return self._config.get("routing", {})

    # -----------------------------------------------------
    # Observability Config
    # -----------------------------------------------------

    def get_observability_config(self):
        """Get full observability configuration."""
        return self._config.get("observability", {})

    def get_telemetry_config(self):
        """Get OpenTelemetry configuration."""
        return self._config.get("observability", {}).get("otel", {})

    def get_langfuse_config(self):
        """Get Langfuse configuration."""
        return self._config.get("observability", {}).get("langfuse", {})

    def get_logging_config(self):
        """Get logging configuration."""
        return self._config.get("observability", {}).get("logging", {})