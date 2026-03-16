"""
McDonald's Customer Support Configuration
Tenant ID: mcdonalds
"""

import os


def get_env(key: str, default=None):
    """Get environment variable with optional default."""
    return os.getenv(key, default)


MCDONALDS_CONFIG = {
    "tenant_id": "mcdonalds",
    "tenant_name": "McDonald's",
    "tenant_type": "qsr",  # Quick Service Restaurant

    # ============================================
    # LLM Configuration
    # ============================================
    "llm": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "max_tokens": 1000
    },

    # ============================================
    # Agent Configuration
    # ============================================
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
        "RAGAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 1
        },
        "EscalationAgent": {
            "enabled": True,
            "prompt_version": "v1",
            "max_retries": 0
        },
        "ResponseAgent": {
            "prompt_version": "v1",
            "max_retries": 0,
            "tone": "friendly",
            "brand_voice": "casual_warm"
        },
        "FallbackAgent": {
            "prompt_version": "v1",
            "max_retries": 0
        }
    },

    # ============================================
    # Intent Classification
    # ============================================
    "intent": {
        "labels": [
            "GREETING",
            "MENU_INQUIRY",
            "ORDER_STATUS",
            "ORDER_ISSUE",
            "REFUND_REQUEST",
            "COMPLAINT",
            "RESTAURANT_INFO",
            "PROMOTION_INQUIRY",
            "PAYMENT_ISSUE",
            "ALLERGY_INFO",
            "FEEDBACK",
            "EMPLOYMENT_INQUIRY",
            "UNKNOWN"
        ],
        "confidence_threshold": 0.7,
        "tool_mapping": {
            "MENU_INQUIRY": "faq_lookup",
            "ORDER_STATUS": "api_tool",
            "ORDER_ISSUE": "api_tool",
            "REFUND_REQUEST": "api_tool",
            "COMPLAINT": "api_tool",
            "RESTAURANT_INFO": "faq_lookup",
            "PROMOTION_INQUIRY": "faq_lookup",
            "PAYMENT_ISSUE": "api_tool",
            "ALLERGY_INFO": "faq_lookup",
            "FEEDBACK": "api_tool"
        }
    },

    # ============================================
    # Sentiment Analysis
    # ============================================
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
            "ANGRY": "I'm really sorry you're upset. Let me help you right away.",
            "FRUSTRATED": "I understand this is frustrating. I'm here to help fix this.",
            "POSITIVE": "Thanks for reaching out! I'm happy to help! :)",
            "NEGATIVE": "I'm sorry to hear that. Let me see what I can do.",
            "NEUTRAL": "Hi there! How can I help you today?"
        }
    },

    # ============================================
    # Business Policies
    # ============================================
    "policy": {
        "enabled": True,
        "policies": {
            "REFUND_REQUEST": {
                "business_action": "PROCESS_REFUND",
                "refund_window_hours": 24,
                "auto_refund_threshold": 5.0,
                "required_fields": ["order_number"],
                "escalate_if_sentiment_above": 0.9,
                "blocked_sentiments": ["ANGRY"],
                "priority": "high",
                "reason": "Refunds within 24 hours for orders under $5"
            },
            "ORDER_ISSUE": {
                "business_action": "REPORT_ISSUE",
                "required_fields": ["order_number", "issue_type"],
                "escalate_if_sentiment_above": 0.8,
                "priority": "high",
                "auto_compensation_threshold": 10.0
            },
            "COMPLAINT": {
                "business_action": "ESCALATE_MANAGER",
                "required_fields": [],
                "escalate_if_sentiment_above": 0.6,
                "priority": "high"
            },
            "MENU_INQUIRY": {
                "business_action": "LOOKUP_FAQ",
                "required_fields": [],
                "escalate_if_sentiment_above": None,
                "priority": "low"
            },
            "ALLERGY_INFO": {
                "business_action": "LOOKUP_FAQ",
                "required_fields": ["allergen"],
                "escalate_if_sentiment_above": None,
                "priority": "high",
                "safety_critical": True
            },
            "PAYMENT_ISSUE": {
                "business_action": "ESCALATE_PAYMENT",
                "required_fields": ["order_number"],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high"
            }
        }
    },

    # ============================================
    # Escalation Configuration
    # ============================================
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
            }
        },
        "tenants": {
            "mcdonalds": {
                "channel": "ticket_system",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH",
                    "NEGATIVE": "MEDIUM"
                },
                "sla_minutes": {
                    "CRITICAL": 15,
                    "HIGH": 30,
                    "MEDIUM": 60
                }
            }
        }
    },

    # ============================================
    # Entity Schema
    # ============================================
    "entities": {
        "mcdonalds": {
            "description": "McDonald's entity schema",
            "fields": {
                "order_number": {
                    "type": "string",
                    "description": "McDonald's order number (e.g., MC-1234567890)",
                    "required_for": ["ORDER_STATUS", "REFUND_REQUEST", "ORDER_ISSUE"]
                },
                "store_number": {
                    "type": "string",
                    "description": "Restaurant/store number (e.g., 12345)",
                    "required_for": ["RESTAURANT_INFO", "COMPLAINT"]
                },
                "item_name": {
                    "type": "string",
                    "description": "Menu item name",
                    "required_for": ["MENU_INQUIRY", "ALLERGY_INFO"]
                },
                "allergen": {
                    "type": "string",
                    "description": "Allergen concern (e.g., peanuts, gluten, dairy)",
                    "required_for": ["ALLERGY_INFO"]
                },
                "transaction_id": {
                    "type": "string",
                    "description": "Payment transaction ID",
                    "required_for": ["PAYMENT_ISSUE", "REFUND_REQUEST"]
                },
                "amount": {
                    "type": "number",
                    "description": "Order amount in USD"
                }
            }
        }
    },

    # ============================================
    # RAG Configuration (Knowledge Base)
    # ============================================
    "rag": {
        "enabled": True,
        "embeddings": {
            "model": "BAAI/bge-base-en-v1.5",
            "dimension": 768
        },
        "vector_store": {
            "type": "qdrant",
            "collections": {
                "faqs": "mcdonalds_faqs",
                "menu": "mcdonalds_menu",
                "policies": "mcdonalds_policies"
            },
            "qdrant_url": get_env("QDRANT_URL", "http://103.180.31.44:8082"),
            "qdrant_api_key": get_env("QDRANT_API_KEY", None)
        }
    },

    # ============================================
    # API Endpoints
    # ============================================
    "api_endpoints": {
        "mcdonalds": {
            "ORDER_STATUS": {
                "url": get_env("MCDONALDS_ORDER_API", "https://api.mcdonalds.com/v1/orders/status"),
                "method": "GET",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": get_env("MCDONALDS_API_KEY", "")
                },
                "timeout": 10,
                "response_mapping": {
                    "status": "order_status",
                    "estimated_time": "estimated_pickup_time",
                    "store_location": "restaurant_address"
                }
            },
            "PROCESS_REFUND": {
                "url": get_env("MCDONALDS_REFUND_API", "https://api.mcdonalds.com/v1/refunds"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-API-Key": get_env("MCDONALDS_API_KEY", "")
                },
                "timeout": 30
            },
            "REPORT_ISSUE": {
                "url": get_env("MCDONALDS_ISSUE_API", "https://api.mcdonalds.com/v1/issues/report"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 20
            },
            "PAYMENT_ISSUE": {
                "url": get_env("MCDONALDS_PAYMENT_API", "https://api.mcdonalds.com/v1/payments/issues"),
                "method": "POST",
                "execution_mode": "async",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            },
            "SUBMIT_FEEDBACK": {
                "url": get_env("MCDONALDS_FEEDBACK_API", "https://api.mcdonalds.com/v1/feedback"),
                "method": "POST",
                "execution_mode": "async",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 10
            }
        }
    },

    # ============================================
    # Tool Configuration
    # ============================================
    "tools": {
        "ticket_creation_tool": {
            "endpoints": {
                "mcdonalds": {
                    "url": get_env("MCDONALDS_TICKET_API", "https://api.mcdonalds.com/support/tickets"),
                    "api_key": get_env("MCDONALDS_TICKET_API_KEY", ""),
                    "queue": "mcdonalds_support",
                    "priority_field": "priority"
                }
            }
        },
        "email_tool": {
            "endpoints": {
                "mcdonalds": {
                    "url": get_env("MCDONALDS_EMAIL_API", "https://api.mcdonalds.com/emails/send"),
                    "api_key": get_env("MCDONALDS_EMAIL_API_KEY", ""),
                    "default_from": "support@mcdonalds.com",
                    "escalation_emails": {
                        "CRITICAL": "escalation-critical@mcdonalds.com",
                        "HIGH": "escalation@mcdonalds.com",
                        "MEDIUM": "support@mcdonalds.com"
                    }
                }
            }
        }
    },

    # ============================================
    # Brand Configuration
    # ============================================
    "brand": {
        "name": "McDonald's",
        "tone": "friendly_casual",
        "greeting": "Hi! Welcome to McDonald's support. How can I help you today?",
        "closing": "Thanks for reaching out! Have a great day! :)",
        "emoji_enabled": True,
        "language": "en",
        "timezone": "America/Chicago"
    },

    # ============================================
    # Routing Configuration
    # ============================================
    "routing": {
        "entry_node": "IntentAgent",
        "nodes": {
            "IntentAgent": {},
            "SentimentAgent": {},
            "ContextBuilderAgent": {},
            "RAGAgent": {},
            "PolicyAgent": {},
            "EscalationAgent": {},
            "ResponseAgent": {},
            "FallbackAgent": {}
        },
        "edges": [
            {
                "from": "IntentAgent",
                "to": "SentimentAgent",
                "condition": {"section": "understanding", "field": "intent.name", "equals": "*"}
            },
            {
                "from": "SentimentAgent",
                "to": "ContextBuilderAgent",
                "condition": {"section": "understanding", "field": "sentiment.label", "equals": "*"}
            },
            {
                "from": "ContextBuilderAgent",
                "to": "RAGAgent",
                "condition": {"section": "context", "field": "entities", "equals": "*"}
            },
            {
                "from": "RAGAgent",
                "to": "PolicyAgent",
                "condition": {"section": "knowledge", "field": "documents", "equals": "*"}
            },
            {
                "from": "PolicyAgent",
                "to": "ResponseAgent",
                "condition": {"section": "policy", "field": "business_action", "equals": "*"}
            },
            {
                "from": "ResponseAgent",
                "to": "EscalationAgent",
                "condition": {"section": "execution", "field": "escalate", "equals": True}
            }
        ],
        "terminal_nodes": ["ResponseAgent", "FallbackAgent"]
    }
}
