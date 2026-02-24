from core.ConfigLoader import ConfigLoader
from core.PromptLoader import PromptLoader
from llms.LLMFactory import LLMFactory
from core.StateManager import StateManager
from core.PatchValidator import PatchValidator
from core.orchestrator import Orchestrator
from core.GraphValidator import GraphValidator
from core.GraphEngine import GraphEngine
from core.ToolRegistry import ToolRegistry

from agents.IntentAgent import IntentAgent
from agents.DecisionAgent import DecisionAgent
from agents.ResponseAgent import ResponseAgent
from agents.ToolExecutionAgent import ToolExecutionAgent
from agents.FallbackAgent import FallbackAgent

from tools.FAQLookupTool import FAQLookupTool

from models.sections import UnderstandingSchema, LifecycleSchema, DecisionSchema, ExecutionSchema


class SystemContainer:
    """
    Holds fully wired system components.
    """

    def __init__(self, graph_engine, orchestrator, state_manager, config_loader, tool_registry):
        self.graph_engine = graph_engine
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.config_loader = config_loader
        self.tool_registry = tool_registry


def bootstrap_system(config: dict) -> SystemContainer:

    # -----------------------------------------------------
    # 1️⃣ Config Loader
    # -----------------------------------------------------
    config_loader = ConfigLoader(config)

    # -----------------------------------------------------
    # 2️⃣ Prompt Loader
    # -----------------------------------------------------
    prompt_loader = PromptLoader()

    # -----------------------------------------------------
    # 3️⃣ LLM Client
    # -----------------------------------------------------
    llm_client = LLMFactory.create(config_loader.get_llm_config())

    # -----------------------------------------------------
    # 4️⃣ State Manager
    # -----------------------------------------------------
    state_manager = StateManager()

    # -----------------------------------------------------
    # 5️⃣ Tool Registry
    # -----------------------------------------------------
    tool_registry = ToolRegistry()
    tool_registry.register(FAQLookupTool())

    # -----------------------------------------------------
    # 5️⃣ Agent Registry
    # -----------------------------------------------------
    agent_registry = {
        "IntentAgent": {
            "allowed_section": "understanding",
            "class": IntentAgent
        },
        "SystemGraphEngine": {
            "allowed_section": "lifecycle",
            "class": None
        },
        "DecisionAgent": {
            "allowed_section": "decision",
            "class": DecisionAgent
        },
        "ToolExecutionAgent": {
            "allowed_section": "execution",
            "class": ToolExecutionAgent
        },
        "ResponseAgent": {
            "allowed_section": "execution",
            "class": ResponseAgent
        },
        "FallbackAgent": {
            "allowed_section": "execution",
            "class": FallbackAgent
        }
    }

    # -----------------------------------------------------
    # 6️⃣ Section Schemas
    # -----------------------------------------------------
    section_schemas = {
        "understanding": UnderstandingSchema,
        "decision": DecisionSchema,
        "execution": ExecutionSchema,
        "lifecycle": LifecycleSchema
    }

    # -----------------------------------------------------
    # 7️⃣ Patch Validator
    # -----------------------------------------------------
    # Create simplified agent authorization map for validator
    agent_authorization = {
        name: config["allowed_section"]
        for name, config in agent_registry.items()
    }

    patch_validator = PatchValidator(
        agent_registry=agent_authorization,
        section_schemas=section_schemas
    )

    # -----------------------------------------------------
    # 8️⃣ Orchestrator
    # -----------------------------------------------------
    orchestrator = Orchestrator(
        state_manager=state_manager,
        patch_validator=patch_validator,
        agent_registry=agent_registry,
        config_loader=config_loader,
        prompt_loader=prompt_loader,
        llm_client=llm_client,
        tool_registry=tool_registry
    )

    # -----------------------------------------------------
    # 9️⃣ Graph Config
    # -----------------------------------------------------
    graph_config = config_loader.get_routing_config()

    # 🔟 Validate Graph (DAG enforcement)
    GraphValidator().validate(graph_config, agent_registry)

    # -----------------------------------------------------
    # 1️⃣1️⃣ Graph Engine
    # -----------------------------------------------------
    graph_engine = GraphEngine(
        graph_config=graph_config,
        orchestrator=orchestrator
    )

    return SystemContainer(
        graph_engine=graph_engine,
        orchestrator=orchestrator,
        state_manager=state_manager,
        config_loader=config_loader,
        tool_registry=tool_registry
    )