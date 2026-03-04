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
        },
        "RAGAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 1
        },
        "WebSearchAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 1
        },
        "EscalationAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 0
        },
        "VannaAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 1
        }
    },

    "vanna": {
        "enabled": True,
        "db_type": "sqlite",
        "dialect": "postgres",
        "connection_string": get_env("DATABASE_URL", "./data/database.db"),
        "schema": """
        -- Orders table
        CREATE TABLE orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT,
            status TEXT,
            amount REAL,
            created_at TIMESTAMP
        );

        -- Payments table
        CREATE TABLE payments (
            payment_id TEXT PRIMARY KEY,
            order_id TEXT,
            status TEXT,
            amount REAL,
            payment_method TEXT,
            created_at TIMESTAMP
        );

        -- Products table
        CREATE TABLE products (
            product_id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        );

        -- Order items table
        CREATE TABLE order_items (
            item_id TEXT PRIMARY KEY,
            order_id TEXT,
            product_id TEXT,
            quantity INTEGER
        );
        """
    },

    "intent": {
        "labels": [
            "GREETING",
            "FAQ_QUERY",
            "SQL_QUERY",
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

    "escalation": {
        "enabled": True,
        "priority_thresholds": {
            "ANGRY": "CRITICAL",
            "FRUSTRATED": "HIGH",
            "NEGATIVE": "MEDIUM"
        },
        "default_channel": "ticket_system",
        "channels": {
            "ticket_system": {
                "description": "Create ticket in support system",
                "tool_mapping": "ticket_creation_tool"
            },
            "email": {
                "description": "Send escalation email",
                "tool_mapping": "email_tool"
            },
            "slack": {
                "description": "Post to Slack channel",
                "tool_mapping": "slack_webhook_tool"
            },
            "webhook": {
                "description": "Call external webhook",
                "tool_mapping": "webhook_tool"
            }
        },
        "tenants": {
            "default": {
                "channel": "ticket_system",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH"
                }
            },
            "amazon": {
                "channel": "ticket_system",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH",
                    "NEGATIVE": "MEDIUM"
                },
                "vip_threshold": "high"
            },
            "flipkart": {
                "channel": "slack",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH"
                }
            },
            "shopify": {
                "channel": "email",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH"
                }
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
    # RAG Configuration (Cascade Retrieval)
    # ============================================
    "rag": {
        "enabled": True,
        "embeddings": {
            "normalize_embeddings": True,
            "device": "cpu",
            "cache_dir": None,
            "stage_1": {
                "model": "BAAI/bge-small-en-v1.5",
                "dimension": 384,
                "description": "Fast, broad search for stage 1"
            },
            "stage_2": {
                "model": "BAAI/bge-base-en-v1.5",
                "dimension": 768,
                "description": "Quality refinement for stage 2"
            }
        },
        "reranker": {
            "enabled": True,
            "type": "bge",  # Options: "bge", "flashrank"
            "model": "BAAI/bge-reranker-large",
            "top_k": 15,
            "batch_size": 16
        },
        "vector_store": {
            "type": "qdrant",
            "collection_prefix": "cs_",
            "distance": "COSINE",
            "qdrant_url": get_env("QDRANT_URL", "http://103.180.31.44:8082"),
            "qdrant_api_key": get_env("QDRANT_API_KEY", None)
        },
        "retrieval": {
            "stage_1": {
                "top_k": 100,
                "score_threshold": 0.3
            },
            "stage_2": {
                "top_k": 50,
                "score_threshold": 0.4
            },
            "reranker": {
                "top_k": 15,
                "score_threshold": 0.5
            }
        },
        "document_ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "default_sources": ["faq", "manual", "sop", "policy"]
        },
        "web_search": {
            "enabled": True,
            "primary_provider": "tavily",
            "max_results": 10,
            "search_depth": "basic",
            "tavily_api_key": get_env("TAVILY_API_KEY"),
            "serper_api_key": get_env("SERPER_API_KEY")
        }
    },

    # ============================================
    # Tool Configuration (Multi-Tenant)
    # ============================================
    "tools": {
        "ticket_creation_tool": {
            "endpoints": {
                "default": {
                    "url": get_env("TICKET_API_URL", "https://api.ticketsystem.com/create"),
                    "api_key": get_env("TICKET_API_KEY", "")
                },
                "amazon": {
                    "url": get_env("AMAZON_TICKET_API_URL", "https://amazon.support.com/api/tickets"),
                    "api_key": get_env("AMAZON_TICKET_API_KEY", "")
                },
                "flipkart": {
                    "url": get_env("FLIPKART_TICKET_API_URL", "https://flipkart.support.com/api/tickets"),
                    "api_key": get_env("FLIPKART_TICKET_API_KEY", "")
                },
                "shopify": {
                    "url": get_env("SHOPIFY_TICKET_API_URL", "https://shopify.support.com/api/tickets"),
                    "api_key": get_env("SHOPIFY_TICKET_API_KEY", "")
                }
            }
        },
        "email_tool": {
            "endpoints": {
                "default": {
                    "url": get_env("EMAIL_API_URL", "https://api.emailservice.com/send"),
                    "api_key": get_env("EMAIL_API_KEY", ""),
                    "default_from": get_env("EMAIL_DEFAULT_FROM", "noreply@example.com")
                },
                "amazon": {
                    "url": get_env("AMAZON_EMAIL_API_URL", "https://ses.amazonaws.com/send"),
                    "api_key": get_env("AMAZON_EMAIL_API_KEY", ""),
                    "default_from": "support@amazon.com"
                },
                "flipkart": {
                    "url": get_env("FLIPKART_EMAIL_API_URL", "https://flipkart.email.com/send"),
                    "api_key": get_env("FLIPKART_EMAIL_API_KEY", ""),
                    "default_from": "support@flipkart.com"
                },
                "shopify": {
                    "url": get_env("SHOPIFY_EMAIL_API_URL", "https://shopify.email.com/send"),
                    "api_key": get_env("SHOPIFY_EMAIL_API_KEY", ""),
                    "default_from": "support@shopify.com"
                }
            }
        },
        "api_tool": {
            "endpoints": {}  # Uses api_endpoints config below
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
            "RAGAgent": {},
            "VannaAgent": {},
            "PolicyAgent": {},
            "DecisionAgent": {},
            "EscalationAgent": {},
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
                "to": "RAGAgent",
                "condition": {
                    "section": "context",
                    "field": "entities",
                    "equals": "*"
                }
            },
            {
                "from": "RAGAgent",
                "to": "PolicyAgent",
                "condition": {
                    "section": "knowledge",
                    "field": "documents",
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
                "to": "EscalationAgent",
                "condition": {
                    "section": "decision",
                    "field": "action",
                    "equals": "ESCALATE"
                }
            },
            # EscalationAgent → ToolExecutionAgent (escalation tool execution)
            {
                "from": "EscalationAgent",
                "to": "ToolExecutionAgent",
                "condition": {
                    "section": "escalation",
                    "field": "business_action",
                    "equals": "*"
                }
            },
            # EscalationAgent → ResponseAgent (escalation complete, inform user)
            {
                "from": "EscalationAgent",
                "to": "ResponseAgent",
                "condition": {
                    "section": "escalation",
                    "field": "status",
                    "equals": "initiated"
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

    # ============================================
    # Observability Configuration
    # ============================================
    "observability": {
        "enabled": True,
        "debug": False,

        # OpenTelemetry (Distributed Tracing)
        "otel": {
            "enabled": True,
            "service_name": "customer-support-resolution",
            "otlp_endpoint": get_env("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            "console_export": get_env("OTEL_CONSOLE_EXPORT", "false") == "true",
            "sample_rate": float(get_env("OTEL_SAMPLE_RATE", "1.0"))
        },

        # Langfuse (LLM Observability)
        "langfuse": {
            "enabled": True,
            "public_key": get_env("LANGFUSE_PUBLIC_KEY"),
            "secret_key": get_env("LANGFUSE_SECRET_KEY"),
            "host": get_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "sample_rate": float(get_env("LANGFUSE_SAMPLE_RATE", "1.0"))
        },

        # Prometheus (Metrics)
        "metrics": {
            "enabled": True,
            "port": int(get_env("METRICS_PORT", "9090")),
            "path": get_env("METRICS_PATH", "/metrics")
        },

        # Logging
        "logging": {
            "level": get_env("LOG_LEVEL", "INFO"),
            "json_output": get_env("LOG_JSON_OUTPUT", "true") == "true"
        }
    }
}