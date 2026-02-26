import os


def get_env(key: str, default=None):
    """Get environment variable with optional default."""
    return os.getenv(key, default)


CONFIG = {
    "llm": {
        "provider": "groq"
    },

    "agents": {
        "IntentAgent": {
            "prompt_version": "v1",
            "max_retries": 1
        },
        "SentimentAgent": {
            "prompt_version": "v1",
            "max_retries": 1
        },
        "PolicyAgent": {
            "prompt_version": "v1",
            "max_retries": 0
        },
        "DecisionAgent": {
            "prompt_version": "v1",
            "max_retries": 0
        },
        "ToolExecutionAgent": {
            "max_retries": 0
        },
        "ResponseAgent": {
            "prompt_version": "v1",
            "max_retries": 0
        },
        "FallbackAgent": {
            "prompt_version": "v1",
            "max_retries": 0
        }
    },

    "intent": {
        "labels": [
            "GREETING",
            "FAQ_QUERY",
            "REFUND_REQUEST",
            "COMPLAINT",
            "DATA_ACCESS",
            "ACCOUNT_UPDATE",
            "UNKNOWN"
        ],
        "confidence_threshold": 0.7,
        "tool_mapping": {
            "FAQ_QUERY": "faq_lookup"
        }
    },

    "sentiment": {
        "labels": [
            "POSITIVE",
            "NEGATIVE",
            "NEUTRAL",
            "FRUSTRATED",
            "ANGRY"
        ],
        "escalation_thresholds": {
            "ANGRY": 0.7,
            "FRUSTRATED": 0.8
        },
        "response_guidelines": {
            "ANGRY": "Show empathy, apologize sincerely, acknowledge their frustration, offer immediate help",
            "FRUSTRATED": "Show empathy, apologize sincerely, acknowledge their frustration, offer immediate help",
            "POSITIVE": "Match their positive tone, thank them, be warm and friendly",
            "NEGATIVE": "Show concern, validate their feelings, offer solutions",
            "NEUTRAL": "Be helpful, concise, and professional"
        }
    },

    "policy": {
        "enabled": True,
        "policies": {
            "angry_customer_policy": {
                "description": "Escalate angry customers regardless of intent",
                "conditions": {
                    "required_intent": [],
                    "blocked_sentiments": ["ANGRY"]
                },
                "actions_on_violation": {
                    "escalate": True,
                    "restrict_response": "I understand you're upset. Let me connect you with a specialist who can help resolve this immediately."
                }
            },
            "refund_policy": {
                "description": "Refund request validation rules",
                "conditions": {
                    "required_intent": ["REFUND_REQUEST"],
                    "max_refund_amount": 500.00,
                    "blocked_sentiments": ["ANGRY"]
                },
                "actions_on_violation": {
                    "escalate": True,
                    "restrict_response": "I'll need to connect you with a specialist for refund requests over $500."
                }
            },
            "complaint_policy": {
                "description": "Formal complaint escalation",
                "conditions": {
                    "required_intent": ["COMPLAINT"]
                },
                "actions_on_violation": {
                    "escalate": True,
                    "restrict_response": "Your complaint has been noted. Let me connect you with a customer specialist."
                }
            },
            "data_access_policy": {
                "description": "Personal data access validation",
                "conditions": {
                    "required_intent": ["DATA_ACCESS", "ACCOUNT_UPDATE"],
                    "require_authentication": True,
                    "blocked_for_tenant_ids": []
                },
                "actions_on_violation": {
                    "escalate": True,
                    "restrict_response": "For security, I cannot process data access requests without proper verification."
                }
            },
            "blocked_tenant_policy": {
                "description": "Blocked tenant check",
                "conditions": {
                    "required_intent": [],
                    "blocked_for_tenant_ids": ["banned_tenant_123"]
                },
                "actions_on_violation": {
                    "escalate": True,
                    "restrict_response": "Service is not available for this account."
                }
            }
        },
        "severity_levels": {
            "LOW": "Minor policy concern, can respond with caveat",
            "MEDIUM": "Policy violation, consider escalation",
            "HIGH": "Significant policy violation, must escalate",
            "CRITICAL": "Severe violation, immediate human intervention required"
        }
    },

    "routing": {
        "entry_node": "IntentAgent",
        "nodes": {
            "IntentAgent": {},
            "SentimentAgent": {},
            "PolicyAgent": {},
            "DecisionAgent": {},
            "ToolExecutionAgent": {},
            "ResponseAgent": {},
            "FallbackAgent": {}
        },
        "edges": [
            {
                "from": "IntentAgent",
                "to": "SentimentAgent",
                "condition": {
                    "section": "understanding",
                    "field": "intent.name",
                    "equals": "*"
                }
            },
            {
                "from": "SentimentAgent",
                "to": "PolicyAgent",
                "condition": {
                    "section": "understanding",
                    "field": "sentiment.label",
                    "equals": "*"
                }
            },
            {
                "from": "PolicyAgent",
                "to": "DecisionAgent",
                "condition": {
                    "section": "policy",
                    "field": "compliant",
                    "equals": "*"
                }
            },
            {
                "from": "DecisionAgent",
                "to": "ToolExecutionAgent",
                "condition": {
                    "section": "decision",
                    "field": "action",
                    "equals": "CALL_TOOL"
                }
            },
            {
                "from": "DecisionAgent",
                "to": "ResponseAgent",
                "condition": {
                    "section": "decision",
                    "field": "action",
                    "equals": "GENERATE_RESPONSE"
                }
            },
            {
                "from": "DecisionAgent",
                "to": "ResponseAgent",
                "condition": {
                    "section": "decision",
                    "field": "action",
                    "equals": "ESCALATE"
                }
            },
            # Tool success → ResponseAgent
            {
                "from": "ToolExecutionAgent",
                "to": "ResponseAgent",
                "condition": {
                    "section": "execution",
                    "field": "tools_called[-1].status",
                    "equals": "success"
                }
            },
            # Tool failure → FallbackAgent
            {
                "from": "ToolExecutionAgent",
                "to": "FallbackAgent",
                "condition": {
                    "section": "execution",
                    "field": "tools_called[-1].status",
                    "equals": "failed"
                }
            }
        ],
        "terminal_nodes": ["ResponseAgent", "FallbackAgent"]
    },

    "observability": {
        "otel": {
            "service_name": "customer-support-resolution",
            "otlp_endpoint": "http://localhost:4317",
            "console_export": True,
            "sample_rate": 1.0
        },
        "langfuse": {
            "public_key": get_env("LANGFUSE_PUBLIC_KEY"),
            "secret_key": get_env("LANGFUSE_SECRET_KEY"),
            "host": "https://cloud.langfuse.com"
        },
        "logging": {
            "level": "INFO",
            "json_output": True
        }
    }
}