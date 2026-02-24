CONFIG = {
    "llm": {
        "provider": "groq"
    },

    "agents": {
        "IntentAgent": {
            "prompt_version": "v1",
            "max_retries": 1
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
            "UNKNOWN"
        ],
        "confidence_threshold": 0.7
    },

    "routing": {
        "entry_node": "IntentAgent",
        "nodes": {
            "IntentAgent": {},
            "DecisionAgent": {},
            "ToolExecutionAgent": {},
            "ResponseAgent": {},
            "FallbackAgent": {}
        },
        "edges": [
            {
                "from": "IntentAgent",
                "to": "DecisionAgent",
                "condition": {
                    "section": "understanding",
                    "field": "intent.name",
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
    }
}