CONFIG = {
    "llm": {
        "provider": "groq"
    },

    "agents": {
        "IntentAgent": {
            "prompt_version": "v1",
            "max_retries": 1
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
            "DecisionAgent": {}
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
            }
        ],
        "terminal_nodes": ["DecisionAgent"]
    }
}