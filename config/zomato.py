"""
Zomato Customer Support Configuration
Tenant ID: zomato
"""

import os


def get_env(key: str, default=None):
    """Get environment variable with optional default."""
    return os.getenv(key, default)


ZOMATO_CONFIG = {
    "tenant_id": "zomato",
    "tenant_name": "Zomato",
    "tenant_type": "food_delivery",  # Food Delivery Platform

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
            "tone": "professional",
            "brand_voice": "helpful_friendly"
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
            "ORDER_STATUS",
            "ORDER_DELAYED",
            "WRONG_ORDER",
            "MISSING_ITEMS",
            "REFUND_REQUEST",
            "COMPLAINT_RESTAURANT",
            "COMPLAINT_DELIVERY",
            "PAYMENT_ISSUE",
            "COUPON_INQUIRY",
            "RESTAURANT_INFO",
            "ACCOUNT_ISSUE",
            "MEMBERSHIP_INQUIRY",
            "FOOD_QUALITY_ISSUE",
            "CANCEL_ORDER",
            "UNKNOWN"
        ],
        "confidence_threshold": 0.7,
        "tool_mapping": {
            "ORDER_STATUS": "api_tool",
            "ORDER_DELAYED": "api_tool",
            "WRONG_ORDER": "api_tool",
            "MISSING_ITEMS": "api_tool",
            "REFUND_REQUEST": "api_tool",
            "COMPLAINT_RESTAURANT": "api_tool",
            "COMPLAINT_DELIVERY": "api_tool",
            "PAYMENT_ISSUE": "api_tool",
            "COUPON_INQUIRY": "faq_lookup",
            "RESTAURANT_INFO": "faq_lookup",
            "ACCOUNT_ISSUE": "api_tool",
            "MEMBERSHIP_INQUIRY": "faq_lookup",
            "CANCEL_ORDER": "api_tool"
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
            "ANGRY": 0.6,
            "FRUSTRATED": 0.75
        },
        "response_guidelines": {
            "ANGRY": "I sincerely apologize for the inconvenience. I understand this is upsetting, and I'm here to help resolve this immediately.",
            "FRUSTRATED": "I'm sorry about this experience. Let me look into this and help you sort it out.",
            "POSITIVE": "Thank you for your patience! Happy to assist you!",
            "NEGATIVE": "I'm sorry to hear about your experience. Let me help you with this.",
            "NEUTRAL": "Hello! I'm here to help. How can I assist you today?"
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
                "refund_window_hours": 48,
                "auto_refund_threshold": 3.0,
                "delay_auto_refund_minutes": 30,
                "required_fields": ["order_id"],
                "escalate_if_sentiment_above": 0.85,
                "blocked_sentiments": ["ANGRY"],
                "priority": "high",
                "reason": "Refunds within 48 hours, auto-refund for delays > 30 min"
            },
            "ORDER_DELAYED": {
                "business_action": "CHECK_DELIVERY",
                "delay_compensation_threshold": 30,
                "compensation_type": "coupon",
                "compensation_amount": 50,
                "required_fields": ["order_id"],
                "escalate_if_sentiment_above": 0.75,
                "priority": "high",
                "sla_response_minutes": 5
            },
            "WRONG_ORDER": {
                "business_action": "REPORT_WRONG_ORDER",
                "required_fields": ["order_id", "wrong_items"],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high",
                "auto_refund": True
            },
            "MISSING_ITEMS": {
                "business_action": "REPORT_MISSING_ITEMS",
                "required_fields": ["order_id", "missing_items"],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high",
                "resolution": "refund_or_redelivery"
            },
            "COMPLAINT_DELIVERY": {
                "business_action": "ESCALATE_DELIVERY_PARTNER",
                "required_fields": ["order_id", "complaint_type"],
                "escalate_if_sentiment_above": 0.65,
                "priority": "high"
            },
            "COMPLAINT_RESTAURANT": {
                "business_action": "ESCALATE_RESTAURANT",
                "required_fields": ["order_id", "restaurant_id"],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high"
            },
            "PAYMENT_ISSUE": {
                "business_action": "ESCALATE_PAYMENT",
                "required_fields": ["order_id", "payment_id"],
                "escalate_if_sentiment_above": 0.75,
                "priority": "critical"
            },
            "CANCEL_ORDER": {
                "business_action": "PROCESS_CANCELLATION",
                "cancellation_window_minutes": 5,
                "required_fields": ["order_id"],
                "escalate_if_sentiment_above": None,
                "priority": "medium",
                "reason": "Can cancel within 5 minutes of placing order"
            },
            "FOOD_QUALITY_ISSUE": {
                "business_action": "REPORT_QUALITY_ISSUE",
                "required_fields": ["order_id", "quality_issue"],
                "escalate_if_sentiment_above": 0.7,
                "priority": "high",
                "photo_required": True
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
            },
            "slack": {
                "description": "Post to internal Slack channel",
                "tool_mapping": "slack_tool"
            }
        },
        "tenants": {
            "zomato": {
                "channel": "ticket_system",
                "priority_mapping": {
                    "ANGRY": "CRITICAL",
                    "FRUSTRATED": "HIGH",
                    "NEGATIVE": "MEDIUM"
                },
                "sla_minutes": {
                    "CRITICAL": 10,
                    "HIGH": 20,
                    "MEDIUM": 45
                },
                "auto_escalate_conditions": {
                    "delayed_order_minutes": 45,
                    "wrong_order_with_angry": True
                }
            }
        }
    },

    # ============================================
    # Entity Schema
    # ============================================
    "entities": {
        "zomato": {
            "description": "Zomato entity schema",
            "fields": {
                "order_id": {
                    "type": "string",
                    "description": "Zomato order ID (e.g., ZO-123456789)",
                    "required_for": ["ORDER_STATUS", "ORDER_DELAYED", "WRONG_ORDER", "MISSING_ITEMS", "REFUND_REQUEST", "CANCEL_ORDER"]
                },
                "restaurant_id": {
                    "type": "string",
                    "description": "Restaurant ID (e.g., 12345)",
                    "required_for": ["COMPLAINT_RESTAURANT", "RESTAURANT_INFO"]
                },
                "delivery_partner_id": {
                    "type": "string",
                    "description": "Delivery partner ID",
                    "required_for": ["COMPLAINT_DELIVERY"]
                },
                "payment_id": {
                    "type": "string",
                    "description": "Payment transaction ID",
                    "required_for": ["PAYMENT_ISSUE", "REFUND_REQUEST"]
                },
                "wrong_items": {
                    "type": "array",
                    "description": "List of wrong items received",
                    "required_for": ["WRONG_ORDER"]
                },
                "missing_items": {
                    "type": "array",
                    "description": "List of missing items",
                    "required_for": ["MISSING_ITEMS"]
                },
                "complaint_type": {
                    "type": "string",
                    "description": "Type of complaint (behavior, hygiene, etc.)",
                    "required_for": ["COMPLAINT_DELIVERY", "COMPLAINT_RESTAURANT"]
                },
                "quality_issue": {
                    "type": "string",
                    "description": "Food quality issue description",
                    "required_for": ["FOOD_QUALITY_ISSUE"]
                },
                "amount": {
                    "type": "number",
                    "description": "Order amount in INR"
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
                "faqs": "zomato_faqs",
                "restaurants": "zomato_restaurants",
                "policies": "zomato_policies"
            },
            "qdrant_url": get_env("QDRANT_URL", "http://103.180.31.44:8082"),
            "qdrant_api_key": get_env("QDRANT_API_KEY", None)
        }
    },

    # ============================================
    # API Endpoints
    # ============================================
    "api_endpoints": {
        "zomato": {
            "ORDER_STATUS": {
                "url": get_env("ZOMATO_ORDER_API", "https://api.zomato.com/v1/orders/status"),
                "method": "GET",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Zomato-API-Key": get_env("ZOMATO_API_KEY", "")
                },
                "timeout": 10,
                "response_mapping": {
                    "status": "order_status",
                    "estimated_delivery": "eta",
                    "delivery_partner": "delivery_agent_name",
                    "live_tracking": "tracking_url"
                }
            },
            "PROCESS_REFUND": {
                "url": get_env("ZOMATO_REFUND_API", "https://api.zomato.com/v1/refunds/process"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Zomato-API-Key": get_env("ZOMATO_API_KEY", "")
                },
                "timeout": 30
            },
            "REPORT_WRONG_ORDER": {
                "url": get_env("ZOMATO_ISSUE_API", "https://api.zomato.com/v1/orders/wrong"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 20
            },
            "REPORT_MISSING_ITEMS": {
                "url": get_env("ZOMATO_ISSUE_API", "https://api.zomato.com/v1/orders/missing"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 20
            },
            "CHECK_DELIVERY": {
                "url": get_env("ZOMATO_DELIVERY_API", "https://api.zomato.com/v1/delivery/status"),
                "method": "GET",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            },
            "PROCESS_CANCELLATION": {
                "url": get_env("ZOMATO_CANCEL_API", "https://api.zomato.com/v1/orders/cancel"),
                "method": "POST",
                "execution_mode": "sync",
                "headers": {
                    "Content-Type": "application/json",
                    "X-Zomato-API-Key": get_env("ZOMATO_API_KEY", "")
                },
                "timeout": 15
            },
            "PAYMENT_ISSUE": {
                "url": get_env("ZOMATO_PAYMENT_API", "https://api.zomato.com/v1/payments/issues"),
                "method": "POST",
                "execution_mode": "async",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 15
            },
            "REPORT_QUALITY_ISSUE": {
                "url": get_env("ZOMATO_QUALITY_API", "https://api.zomato.com/v1/quality/report"),
                "method": "POST",
                "execution_mode": "async",
                "headers": {
                    "Content-Type": "application/json"
                },
                "timeout": 20
            }
        }
    },

    # ============================================
    # Tool Configuration
    # ============================================
    "tools": {
        "ticket_creation_tool": {
            "endpoints": {
                "zomato": {
                    "url": get_env("ZOMATO_TICKET_API", "https://api.zomato.com/support/tickets"),
                    "api_key": get_env("ZOMATO_TICKET_API_KEY", ""),
                    "queue": "zomato_support",
                    "priority_field": "priority",
                    "auto_assign": {
                        "CRITICAL": "vip_team",
                        "HIGH": "priority_team",
                        "MEDIUM": "general_team"
                    }
                }
            }
        },
        "email_tool": {
            "endpoints": {
                "zomato": {
                    "url": get_env("ZOMATO_EMAIL_API", "https://api.zomato.com/emails/send"),
                    "api_key": get_env("ZOMATO_EMAIL_API_KEY", ""),
                    "default_from": "support@zomato.com",
                    "escalation_emails": {
                        "CRITICAL": "escalation-critical@zomato.com",
                        "HIGH": "escalation@zomato.com",
                        "MEDIUM": "support@zomato.com"
                    }
                }
            }
        },
        "slack_tool": {
            "endpoints": {
                "zomato": {
                    "webhook_url": get_env("ZOMATO_SLACK_WEBHOOK", ""),
                    "channels": {
                        "CRITICAL": "#support-critical",
                        "HIGH": "#support-urgent",
                        "MEDIUM": "#support-queue"
                    }
                }
            }
        }
    },

    # ============================================
    # Brand Configuration
    # ============================================
    "brand": {
        "name": "Zomato",
        "tone": "professional_friendly",
        "greeting": "Hello! Welcome to Zomato Support. How can I help you with your order today?",
        "closing": "Thank you for contacting Zomato. Have a great day and happy ordering!",
        "emoji_enabled": True,
        "language": "en",
        "timezone": "Asia/Kolkata"
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
