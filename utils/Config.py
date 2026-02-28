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
        "ContextBuilderAgent": {
            "prompt_version": "v1",
            "max_retries": 2
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
            "FAQ_QUERY": "faq_lookup",
            "LOOKUP_FAQ": "faq_lookup",
            "PROCESS_REFUND": "api_tool",
            "PROCESS_COMPLAINT": "api_tool",
            "VERIFY_DATA_ACCESS": "api_tool",
            "PROCESS_ACCOUNT_UPDATE": "api_tool"
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
            "REFUND_REQUEST": {
                "business_action": "PROCESS_REFUND",
                "required_fields": ["order_id"],
                "escalate_if_sentiment_above": 0.9,
                "blocked_sentiments": ["ANGRY"],
                "priority": "high"
            },
            "COMPLAINT": {
                "business_action": "PROCESS_COMPLAINT",
                "required_fields": [],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high"
            },
            "DATA_ACCESS": {
                "business_action": "VERIFY_DATA_ACCESS",
                "required_fields": ["account_id", "verification_code"],
                "escalate_if_sentiment_above": None,
                "priority": "normal"
            },
            "ACCOUNT_UPDATE": {
                "business_action": "PROCESS_ACCOUNT_UPDATE",
                "required_fields": ["account_id"],
                "escalate_if_sentiment_above": 0.85,
                "priority": "normal"
            },
            "FAQ_QUERY": {
                "business_action": "LOOKUP_FAQ",
                "required_fields": [],
                "escalate_if_sentiment_above": None,
                "priority": "low"
            }
        }
    },

    "entities": {
        "default": {
            "description": "Default entity schema when tenant not specified",
            "fields": {
                "order_id": {"type": "string", "description": "Order or transaction ID"},
                "amount": {"type": "number", "description": "Monetary amount"},
                "currency": {"type": "string", "description": "Currency code (USD, EUR, INR, etc.)"}
            }
        },
        "amazon": {
            "description": "Amazon tenant entity schema",
            "fields": {
                "order_id": {"type": "string", "description": "Amazon order ID (e.g., 123-4567890-1234567)"},
                "payment_id": {"type": "string", "description": "Payment transaction ID"},
                "amount": {"type": "number", "description": "Refund or purchase amount"},
                "currency": {"type": "string", "description": "Currency code"}
            }
        },
        "flipkart": {
            "description": "Flipkart tenant entity schema",
            "fields": {
                "invoice_id": {"type": "string", "description": "Flipkart invoice ID"},
                "amount": {"type": "number", "description": "Amount in rupees"},
                "currency": {"type": "string", "description": "INR"}
            }
        },
        "shopify": {
            "description": "Shopify merchant entity schema",
            "fields": {
                "order_number": {"type": "string", "description": "Shopify order number"},
                "customer_email": {"type": "string", "description": "Customer email address"},
                "amount": {"type": "number", "description": "Order amount"},
                "currency": {"type": "string", "description": "Currency code"}
            }
        }
    },

    # ============================================
    # API Tool Configuration (Multi-Tenant Endpoints)
    # ============================================
    "api_endpoints": {
        "default": {
            "PROCESS_REFUND": {
                "url": "https://api.example.com/refunds",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": get_env("DEFAULT_API_KEY", "demo_key")
                },
                "timeout": 30,
                "retry_attempts": 2,
                "retry_delay_seconds": 1,
                "response_mapping": {
                    "refund_status": "status",
                    "refund_id": "id",
                    "estimated_days": "processing_time_days"
                }
            },
            "PROCESS_COMPLAINT": {
                "url": "https://api.example.com/tickets",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            },
            "VERIFY_DATA_ACCESS": {
                "url": "https://api.example.com/data/verify",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 10
            },
            "PROCESS_ACCOUNT_UPDATE": {
                "url": "https://api.example.com/account/update",
                "method": "PUT",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 20
            }
        },
        "amazon": {
            "PROCESS_REFUND": {
                "url": "https://api.amazon.com/marketplace/refunds/v1",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Amz-Target": "com.amazon.esd.retail.EsdRetailService.RefundRequest"
                },
                "timeout": 30
            },
            "PROCESS_COMPLAINT": {
                "url": "https://api.amazon.com/support/tickets",
                "method": "POST",
                "execution_mode": "async",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            }
        },
        "flipkart": {
            "PROCESS_REFUND": {
                "url": "https://api.flipkart.com/refunds/v2/process",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-FK-API-Key": get_env("FLIPKART_API_KEY", "")
                },
                "timeout": 25
            }
        },
        "shopify": {
            "PROCESS_REFUND": {
                "url": "https://shopify.com/api/refunds",
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Shopify-Access-Token": get_env("SHOPIFY_ACCESS_TOKEN", "")
                },
                "timeout": 20
            },
            "PROCESS_ACCOUNT_UPDATE": {
                "url": "https://shopify.com/api/customer/update",
                "method": "PUT",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            }
        }
    },

    "routing": {
        "entry_node": "IntentAgent",
        "nodes": {
            "IntentAgent": {},
            "SentimentAgent": {},
            "ContextBuilderAgent": {},
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
                "to": "ContextBuilderAgent",
                "condition": {
                    "section": "understanding",
                    "field": "sentiment.label",
                    "equals": "*"
                }
            },
            {
                "from": "ContextBuilderAgent",
                "to": "PolicyAgent",
                "condition": {
                    "section": "context",
                    "field": "entities",
                    "equals": "*"
                }
            },
            {
                "from": "PolicyAgent",
                "to": "DecisionAgent",
                "condition": {
                    "section": "policy",
                    "field": "business_action",
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
            },
            # Tool pending (async) → ResponseAgent
            {
                "from": "ToolExecutionAgent",
                "to": "ResponseAgent",
                "condition": {
                    "section": "execution",
                    "field": "tools_called[-1].status",
                    "equals": "pending"
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