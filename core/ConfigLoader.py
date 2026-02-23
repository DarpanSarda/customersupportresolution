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

    # -----------------------------------------------------
    # Routing Config
    # -----------------------------------------------------

    def get_routing_config(self):
        return self._config.get("routing", {})