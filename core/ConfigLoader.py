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
        """Get specific policy by name (legacy method)."""
        return self.get_policies().get(policy_name, {})

    def get_policy_for_intent(self, intent_name: str):
        """Get policy configuration for a specific intent (keyed by intent name)."""
        return self.get_policies().get(intent_name, {})

    # -----------------------------------------------------
    # Entity Config
    # -----------------------------------------------------

    def get_entity_schema(self, tenant_id: str = "default"):
        """Get entity schema for a specific tenant."""
        entities = self._config.get("entities", {})
        return entities.get(tenant_id, entities.get("default", {}))

    # -----------------------------------------------------
    # RAG Config
    # -----------------------------------------------------

    def is_rag_enabled(self):
        """Check if RAG retrieval is enabled."""
        return self._config.get("rag", {}).get("enabled", True)

    def get_rag_config(self):
        """Get full RAG configuration."""
        return self._config.get("rag", {})

    def get_embeddings_config(self):
        """Get embeddings configuration."""
        return self._config.get("rag", {}).get("embeddings", {})

    def get_stage_embedding_config(self, stage: str):
        """Get embedding config for specific stage (stage_1 or stage_2)."""
        embeddings = self.get_embeddings_config()
        return embeddings.get(stage, {})

    def get_reranker_config(self):
        """Get reranker configuration."""
        return self._config.get("rag", {}).get("reranker", {})

    def is_reranker_enabled(self):
        """Check if reranker is enabled."""
        return self._config.get("rag", {}).get("reranker", {}).get("enabled", True)

    def get_vector_store_config(self):
        """Get vector store configuration."""
        return self._config.get("rag", {}).get("vector_store", {})

    def get_retrieval_config(self):
        """Get retrieval configuration."""
        return self._config.get("rag", {}).get("retrieval", {})

    def get_stage_retrieval_config(self, stage: str):
        """Get retrieval config for specific stage."""
        retrieval = self.get_retrieval_config()
        return retrieval.get(stage, {})

    def is_web_search_enabled(self):
        """Check if web search is enabled."""
        return self._config.get("rag", {}).get("web_search", {}).get("enabled", False)

    def get_web_search_config(self):
        """Get web search configuration."""
        return self._config.get("rag", {}).get("web_search", {})

    # -----------------------------------------------------
    # Routing Config
    # -----------------------------------------------------

    def get_routing_config(self):
        return self._config.get("routing", {})

    # -----------------------------------------------------
    # Observability Config
    # -----------------------------------------------------

    def is_observability_enabled(self):
        """Check if observability is enabled."""
        return self._config.get("observability", {}).get("enabled", True)

    def get_observability_config(self):
        """Get full observability configuration."""
        return self._config.get("observability", {})

    def get_telemetry_config(self):
        """Get OpenTelemetry configuration."""
        return self._config.get("observability", {}).get("otel", {})

    def get_langfuse_config(self):
        """Get Langfuse configuration."""
        return self._config.get("observability", {}).get("langfuse", {})

    def get_metrics_config(self):
        """Get Prometheus metrics configuration."""
        return self._config.get("observability", {}).get("metrics", {})

    def is_metrics_enabled(self):
        """Check if metrics collection is enabled."""
        return self._config.get("observability", {}).get("metrics", {}).get("enabled", True)

    def is_otel_enabled(self):
        """Check if OpenTelemetry tracing is enabled."""
        return self._config.get("observability", {}).get("otel", {}).get("enabled", True)

    def is_langfuse_enabled(self):
        """Check if Langfuse tracing is enabled."""
        return self._config.get("observability", {}).get("langfuse", {}).get("enabled", True)

    def get_logging_config(self):
        """Get logging configuration."""
        return self._config.get("observability", {}).get("logging", {})

    def is_debug_mode(self):
        """Check if debug mode is enabled."""
        return self._config.get("observability", {}).get("debug", False)

    # -----------------------------------------------------
    # Escalation Config
    # -----------------------------------------------------

    def is_escalation_enabled(self):
        """Check if escalation is enabled."""
        return self._config.get("escalation", {}).get("enabled", True)

    def get_escalation_config(self):
        """Get full escalation configuration."""
        return self._config.get("escalation", {})

    def get_escalation_priority_thresholds(self):
        """Get sentiment-to-priority mapping for escalation."""
        return self._config.get("escalation", {}).get("priority_thresholds", {})

    def get_escalation_channel(self):
        """Get default escalation channel."""
        return self._config.get("escalation", {}).get("default_channel", "ticket_system")

    def get_tenant_escalation_config(self, tenant_id: str = "default"):
        """Get tenant-specific escalation configuration."""
        tenants = self._config.get("escalation", {}).get("tenants", {})
        return tenants.get(tenant_id, tenants.get("default", {}))

    # -----------------------------------------------------
    # Vanna Config
    # -----------------------------------------------------

    def is_vanna_enabled(self):
        """Check if Vanna text-to-SQL is enabled."""
        return self._config.get("vanna", {}).get("enabled", True)

    def get_vanna_config(self):
        """Get full Vanna configuration."""
        return self._config.get("vanna", {})