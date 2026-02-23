# bootstrap.py

from core.ConfigLoader import ConfigLoader
from core.PromptLoader import PromptLoader
from llms.LLMFactory import LLMFactory
from core.StateManager import StateManager
from core.PatchValidator import PatchValidator
from core.orchestrator import Orchestrator
from core.GraphValidator import GraphValidator
from core.GraphEngine import GraphEngine

from agents.IntentAgent import IntentAgent

from models.sections import UnderstandingSchema, LifecycleSchema

# -----------------------------------------------------
# CONFIG
# -----------------------------------------------------

CONFIG = {
    "llm": {
        "provider": "groq",  # or openrouter
        "model": "llama3-70b-8192",
        "api_key": "YOUR_API_KEY"
    },
    "agents": {
        "IntentAgent": {
            "prompt_version": "v1",
            "max_retries": 1
        }
    },
    "intent": {
        "labels": ["GREETING", "FAQ_QUERY", "UNKNOWN"],
        "confidence_threshold": 0.7
    },
    "routing": {
        "entry_node": "IntentAgent",
        "nodes": {
            "IntentAgent": {}
        },
        "edges": [],
        "terminal_nodes": ["IntentAgent"]
    }
}


def bootstrap_system():

    # 1️⃣ Config
    config_loader = ConfigLoader(CONFIG)

    # 2️⃣ Prompt Loader
    prompt_loader = PromptLoader()

    # 3️⃣ LLM Client
    llm_client = LLMFactory.create(config_loader.get_llm_config())

    # 4️⃣ State Manager
    state_manager = StateManager()

    # 5️⃣ Agent Registry
    agent_registry = {
        "IntentAgent": {
            "allowed_section": "understanding",
            "class": IntentAgent
        },
        "SystemGraphEngine": {
            "allowed_section": "lifecycle",
            "class": None
        }
    }

    # 6️⃣ Section Schemas
    section_schemas = {
        "understanding": UnderstandingSchema,
        "lifecycle": LifecycleSchema
    }

    # 7️⃣ Patch Validator
    patch_validator = PatchValidator(
        agent_registry=agent_registry,
        section_schemas=section_schemas
    )

    # 8️⃣ Orchestrator
    orchestrator = Orchestrator(
        state_manager=state_manager,
        patch_validator=patch_validator,
        agent_registry=agent_registry,
        config_loader=config_loader,
        prompt_loader=prompt_loader,
        llm_client=llm_client
    )

    # 9️⃣ Graph Config
    graph_config = config_loader.get_routing_config()

    # 🔟 Validate Graph
    GraphValidator().validate(graph_config, agent_registry)

    # 1️⃣1️⃣ Graph Engine
    graph_engine = GraphEngine(
        graph_config=graph_config,
        orchestrator=orchestrator
    )

    return graph_engine