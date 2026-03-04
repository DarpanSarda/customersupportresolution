from core.ConfigLoader import ConfigLoader
from core.PromptLoader import PromptLoader
from llms.LLMFactory import LLMFactory
from core.StateManager import StateManager
from core.PatchValidator import PatchValidator
from core.orchestrator import Orchestrator
from core.GraphValidator import GraphValidator
from core.GraphEngine import GraphEngine
from core.ToolRegistry import ToolRegistry
from observability.structured_logger import StructuredLogger
from observability import (
    LangfuseTracer,
    OTELTracer,
    MetricsCollector,
    TracingContext
)

from agents.IntentAgent import IntentAgent
from agents.SentimentAgent import SentimentAgent
from agents.ContextBuilderAgent import ContextBuilderAgent
from agents.RAGAgent import RAGAgent
from agents.WebSearchAgent import WebSearchAgent
from agents.PolicyAgent import PolicyAgent
from agents.DecisionAgent import DecisionAgent
from agents.EscalationAgent import EscalationAgent
from agents.VannaAgent import VannaAgent
from agents.ResponseAgent import ResponseAgent
from agents.ToolExecutionAgent import ToolExecutionAgent
from agents.FallbackAgent import FallbackAgent

from tools.FAQLookupTool import FAQLookupTool
from tools.ApiTool import ApiTool
from tools.TicketCreationTool import TicketCreationTool
from tools.EmailTool import EmailTool

from models.sections import UnderstandingSchema, LifecycleSchema, DecisionSchema, ExecutionSchema, PolicySchema, ContextSchema, RAGSchema, WebSearchSchema, EscalationSchema, VannaSchema


def _initialize_rag_collections(qdrant_store, embeddings_config, tenant_id: str = "default"):
    """Initialize empty RAG collections for cascade retrieval.

    Creates collections for stage_1 and stage_2 if they don't exist.

    Args:
        qdrant_store: QdrantVectorStore instance
        embeddings_config: Embeddings configuration
        tenant_id: Tenant ID for collection naming
    """
    from qdrant_client.models import Distance, VectorParams
    from qdrant_client.http.exceptions import UnexpectedResponse

    client = qdrant_store.client

    # Get collection names
    prefix = "cs_"
    stage_1_col = f"{prefix}{tenant_id}_stage_1"
    stage_2_col = f"{prefix}{tenant_id}_stage_2"

    # Get vector sizes from config
    stage_1_size = embeddings_config.get("stage_1", {}).get("dimension", 384)
    stage_2_size = embeddings_config.get("stage_2", {}).get("dimension", 768)

    # Create stage_1 collection if not exists
    try:
        client.create_collection(
            collection_name=stage_1_col,
            vectors_config=VectorParams(
                size=stage_1_size,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {stage_1_col}")
    except UnexpectedResponse as e:
        # Collection already exists (status code 409 = Conflict)
        if e.status_code == 409:
            print(f"Collection already exists: {stage_1_col}")
        else:
            raise
    except Exception as e:
        print(f"Error creating collection {stage_1_col}: {e}")

    # Create stage_2 collection if not exists
    try:
        client.create_collection(
            collection_name=stage_2_col,
            vectors_config=VectorParams(
                size=stage_2_size,
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {stage_2_col}")
    except UnexpectedResponse as e:
        # Collection already exists (status code 409 = Conflict)
        if e.status_code == 409:
            print(f"Collection already exists: {stage_2_col}")
        else:
            raise
    except Exception as e:
        print(f"Error creating collection {stage_2_col}: {e}")


class SystemContainer:
    """
    Holds fully wired system components.
    """

    def __init__(self, graph_engine, orchestrator, state_manager, config_loader, tool_registry, logger=None, tracing_context=None):
        self.graph_engine = graph_engine
        self.orchestrator = orchestrator
        self.state_manager = state_manager
        self.config_loader = config_loader
        self.tool_registry = tool_registry
        self.logger = logger
        self.tracing_context = tracing_context


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
    llm_config = config_loader.get_llm_config()
    llm_client = LLMFactory.create(llm_config)

    # -----------------------------------------------------
    # 4️⃣ State Manager
    # -----------------------------------------------------
    state_manager = StateManager()

    # -----------------------------------------------------
    # 5️⃣ RAG Components (Cascade Retrieval)
    # -----------------------------------------------------
    embeddings = None
    vector_store = None
    reranker = None

    if config_loader.is_rag_enabled():
        from utils.Embeddings import CascadeEmbeddings
        from utils.VectorStore import CascadeVectorStore, QdrantVectorStore
        from utils.Reranker import BGEReranker, FlashRankReranker

        # Initialize cascade embeddings
        embeddings = CascadeEmbeddings.from_config(
            config=config_loader.get_embeddings_config(),
            device=config_loader.get_embeddings_config().get("device", "cpu")
        )

        # Initialize vector store
        vs_config = config_loader.get_vector_store_config()
        if vs_config.get("type") == "qdrant":
            qdrant_store = QdrantVectorStore(
                url=vs_config.get("qdrant_url", "http://103.180.31.44:8082"),
                api_key=vs_config.get("qdrant_api_key"),
                timeout=vs_config.get("timeout", 60),
                prefer_grpc=vs_config.get("prefer_grpc", False)
            )
            vector_store = CascadeVectorStore(vector_store=qdrant_store)

            # Initialize empty collections for cascade retrieval
            # Create for common tenant IDs
            tenant_ids = ["default", "test", "amazon", "flipkart", "shopify"]
            for tenant_id in tenant_ids:
                _initialize_rag_collections(
                    qdrant_store,
                    config_loader.get_embeddings_config(),
                    tenant_id=tenant_id
                )

        # Initialize reranker if enabled
        if config_loader.is_reranker_enabled():
            reranker_config = config_loader.get_reranker_config()
            reranker_type = reranker_config.get("type", "bge")
            if reranker_type == "bge":
                reranker = BGEReranker(
                    model_name=reranker_config.get("model", "BAAI/bge-reranker-large")
                )
            elif reranker_type == "flashrank":
                reranker = FlashRankReranker()

    # -----------------------------------------------------
    # 6️⃣ Tool Registry
    # -----------------------------------------------------
    tool_registry_config = {
        "tool_mapping": config.get("intent", {}).get("tool_mapping", {}),
        "endpoints": config.get("api_endpoints", {})
    }

    tool_registry = ToolRegistry(config=tool_registry_config)

    # Register tools
    tool_registry.register(FAQLookupTool())
    tool_registry.register(ApiTool(config=tool_registry_config))

    # Register escalation tools
    tools_config = config.get("tools", {})
    ticket_tool_config = tools_config.get("ticket_creation_tool", {})
    email_tool_config = tools_config.get("email_tool", {})
    tool_registry.register(TicketCreationTool(config=ticket_tool_config))
    tool_registry.register(EmailTool(config=email_tool_config))

    # -----------------------------------------------------
    # 7️⃣ Agent Registry
    # -----------------------------------------------------
    agent_registry = {
        "IntentAgent": {
            "allowed_section": "understanding",
            "class": IntentAgent
        },
        "SentimentAgent": {
            "allowed_section": "understanding",
            "class": SentimentAgent
        },
        "ContextBuilderAgent": {
            "allowed_section": "context",
            "class": ContextBuilderAgent
        },
        "RAGAgent": {
            "allowed_section": "knowledge",
            "class": RAGAgent,
            "embeddings": embeddings,
            "vector_store": vector_store,
            "reranker": reranker
        },
        "WebSearchAgent": {
            "allowed_section": "web_search",
            "class": WebSearchAgent,
            "config_loader": config_loader,
            "web_search_config": config_loader.get_web_search_config()
        },
        "PolicyAgent": {
            "allowed_section": "policy",
            "class": PolicyAgent
        },
        "SystemGraphEngine": {
            "allowed_section": "lifecycle",
            "class": None
        },
        "DecisionAgent": {
            "allowed_section": "decision",
            "class": DecisionAgent
        },
        "EscalationAgent": {
            "allowed_section": "escalation",
            "class": EscalationAgent,
            "config_loader": config_loader,
            "escalation_config": config_loader.get_escalation_config()
        },
        "VannaAgent": {
            "allowed_section": "vanna",
            "class": VannaAgent,
            "llm_client": llm_client,
            "vanna_config": config_loader.get_vanna_config()
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
    # 8️⃣ Section Schemas
    # -----------------------------------------------------
    section_schemas = {
        "understanding": UnderstandingSchema,
        "context": ContextSchema,
        "knowledge": RAGSchema,
        "web_search": WebSearchSchema,
        "policy": PolicySchema,
        "decision": DecisionSchema,
        "escalation": EscalationSchema,
        "vanna": VannaSchema,
        "execution": ExecutionSchema,
        "lifecycle": LifecycleSchema
    }

    # -----------------------------------------------------
    # 9️⃣ Patch Validator
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
    # 1️⃣0️⃣ Orchestrator
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
    # 1️⃣1️⃣ Graph Config
    # -----------------------------------------------------
    graph_config = config_loader.get_routing_config()

    # 1️⃣2️⃣ Validate Graph (DAG enforcement)
    GraphValidator().validate(graph_config, agent_registry)

    # -----------------------------------------------------
    # 1️⃣3️⃣ Structured Logger
    # -----------------------------------------------------
    structured_logger = None
    logging_config = config_loader.get_logging_config()
    if logging_config:
        try:
            structured_logger = StructuredLogger(
                service_name="customer-support-resolution",
                level=logging_config.get("level", "INFO"),
                json_output=logging_config.get("json_output", True)
            )
        except Exception:
            structured_logger = None

    # -----------------------------------------------------
    # 1️⃣4️⃣ Observability Tracers
    # -----------------------------------------------------
    langfuse_tracer = None
    otel_tracer = None
    metrics_collector = None
    tracing_context = None

    if config_loader.is_observability_enabled():
        # Initialize Langfuse
        if config_loader.is_langfuse_enabled():
            langfuse_config = config_loader.get_langfuse_config()
            langfuse_tracer = LangfuseTracer(
                public_key=langfuse_config.get("public_key"),
                secret_key=langfuse_config.get("secret_key"),
                host=langfuse_config.get("host"),
                enabled=langfuse_config.get("enabled", True),
                sample_rate=langfuse_config.get("sample_rate", 1.0),
                debug=config_loader.is_debug_mode()
            )

        # Initialize OTEL
        if config_loader.is_otel_enabled():
            otel_config = config_loader.get_telemetry_config()
            otel_tracer = OTELTracer(
                service_name=otel_config.get("service_name", "customer-support-resolution"),
                otlp_endpoint=otel_config.get("otlp_endpoint", "http://localhost:4317"),
                console_export=otel_config.get("console_export", False),
                sample_rate=otel_config.get("sample_rate", 1.0),
                enabled=otel_config.get("enabled", True),
                debug=config_loader.is_debug_mode()
            )

        # Initialize Metrics
        if config_loader.is_metrics_enabled():
            metrics_config = config_loader.get_metrics_config()
            metrics_collector = MetricsCollector(
                enabled=metrics_config.get("enabled", True),
                debug=config_loader.is_debug_mode()
            )

        # Create unified tracing context
        tracing_context = TracingContext(
            langfuse=langfuse_tracer,
            otel=otel_tracer,
            metrics=metrics_collector
        )

    # -----------------------------------------------------
    # 1️⃣5️⃣ Graph Engine
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
        tool_registry=tool_registry,
        logger=structured_logger,
        tracing_context=tracing_context
    )
