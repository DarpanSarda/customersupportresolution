"""
ChatService - Service for handling chat requests.

Handles Orchestrator initialization and request processing.
Following the Agent Contract pattern with shared state management.
"""

from fastapi import HTTPException
from llms.BaseLLM import LLMConfig
from llms.LLMFactory import LLMFactory
from utils.DBManager import get_db_manager
from services.ConfigService import ConfigService
from utils.encryption import decrypt
from schemas.chat import ChatRequest, ChatResponse
from core.Orchestrator import Orchestrator, AgentFactory
import os


class ChatService:
    """
    Service for processing chat requests.

    Manages IntentAgent lifecycle and database integration.
    """

    # Common provider name typos and corrections
    PROVIDER_NORMALIZATION = {
        "openai-compitable": "openai-compatible",
        "openai_compatible": "openai-compatible",
        "openaicompatible": "openai-compatible",
    }

    def __init__(self, db_manager):
        """
        Initialize ChatService.

        Args:
            db_manager: DBManager instance
        """
        self.db = db_manager
        self.config_service = ConfigService(db_manager)

    def _normalize_provider(self, provider: str) -> str:
        """
        Normalize provider name to handle common typos and variations.

        Args:
            provider: Raw provider name from database

        Returns:
            Normalized provider name
        """
        provider_lower = provider.lower().strip()
        return self.PROVIDER_NORMALIZATION.get(provider_lower, provider_lower)

    async def get_llm_client(
        self,
        chatbot_id: str
    ):
        """
        Fetch LLM credentials from database and create LLM client.

        Args:
            chatbot_id: Chatbot identifier

        Returns:
            LLM client instance

        Raises:
            HTTPException: If credentials not found
        """
        # Fetch LLM credentials from database
        llm_creds = await self.db.fetch_one(
            table="llm_credentials",
            filters={
                "chatbot_id": chatbot_id
            }
        )
        print(f"Fetched LLM credentials for chatbot_id: {chatbot_id}{llm_creds}")

        if not llm_creds:
            raise HTTPException(
                status_code=500,
                detail=f"LLM credentials not found for chatbot '{chatbot_id}'"
            )

        # Extract provider from database response and normalize it
        provider_raw = llm_creds.get("provider", "openai")
        provider = self._normalize_provider(provider_raw)
        print(f"[ChatService] Provider normalized: '{provider_raw}' -> '{provider}'")

        # Decrypt the API key using AES-256-GCM
        api_key = decrypt(
            encrypted_data=llm_creds["api_key_encrypted"],
            iv_data=llm_creds["iv"],
            tag_data=llm_creds["tag"]
        )

        # Check for custom base_url in database
        base_url = llm_creds.get("base_url")

        # If there's a custom base_url with provider='openai', use 'openai-compatible' instead
        # This handles cases where the database has Groq/other providers with 'openai' provider
        actual_provider = provider
        if provider == "openai" and base_url:
            # Use the openai-compatible manager for custom base URLs
            actual_provider = "openai-compatible"

        # Prepare extra_params if base_url exists
        extra_params = None
        if base_url:
            extra_params = {"base_url": base_url}

        config = LLMConfig(
            provider=actual_provider,
            api_key=api_key,
            model=llm_creds.get("model", "gpt-4o-mini"),
            temperature=llm_creds.get("temperature", 0.1),
            max_tokens=llm_creds.get("max_tokens", 200),
            top_p=llm_creds.get("top_p", 1.0),
            frequency_penalty=llm_creds.get("frequency_penalty", 0.0),
            presence_penalty=llm_creds.get("presence_penalty", 0.0),
            extra_params=extra_params
        )

        return LLMFactory.create(actual_provider, config)

    async def _create_agent_factory(self, llm_client) -> AgentFactory:
        """
        Create AgentFactory for instantiating agents.

        Args:
            llm_client: LLM client instance

        Returns:
            AgentFactory instance
        """
        return AgentFactory(
            llm_client=llm_client,
            config_service=self.config_service
        )

    async def _create_tool_registry(self):
        """
        Create ToolRegistry with available tools.

        Returns:
            ToolRegistry instance
        """
        from core.ToolRegistry import ToolRegistry
        from tools.FAQTool import FAQTool
        from tools.ApiTool import ApiTool
        from models.tool import ToolConfig as ToolConfigModel

        # Create tool registry
        registry = ToolRegistry()

        # Register FAQTool
        faq_tool = FAQTool()
        print(f"[DEBUG] Creating FAQTool, name={faq_tool.get_name()}")
        registry.register_base_tool(faq_tool)
        print(f"[DEBUG] Registered tool: FAQTool with name={faq_tool.get_name()}")

        # Register ApiTool with default config
        api_tool_config = ToolConfigModel(
            url="",
            method="POST",
            timeout_seconds=30,
            retry_attempts=2,
            headers={}
        )
        api_tool = ApiTool(config=api_tool_config)
        print(f"[DEBUG] Creating ApiTool, name={api_tool.get_name()}")
        registry.register_base_tool(api_tool)
        print(f"[DEBUG] Registered tool: ApiTool with name={api_tool.get_name()}")

        # Set up agent permissions
        # ResponseAgent can use FAQTool
        registry.set_agent_permissions("ResponseAgent", ["FAQTool"])
        print(f"[DEBUG] Set permissions for ResponseAgent: {registry.get_agent_tools('ResponseAgent')}")

        # RAGRetrievalAgent can use FAQTool
        registry.set_agent_permissions("RAGRetrievalAgent", ["FAQTool"])
        print(f"[DEBUG] Set permissions for RAGRetrievalAgent: {registry.get_agent_tools('RAGRetrievalAgent')}")

        print(f"[DEBUG] Tool registry created with {len(registry.list_tools())} tools")
        print(f"[DEBUG] Available tools: {registry.list_tools()}")
        print(f"[DEBUG] All permissions: {registry.get_permissions_info()}")

        return registry

    async def _get_orchestrator_config(self, tenant_id: str) -> dict:
        """
        Get configuration for orchestrator.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Configuration dict with intents, RAG settings, etc.
        """
        try:
            intents = await self.config_service.get_intents(tenant_id=tenant_id)
            return {
                "intents": intents,
                "rag_top_k": 3,
                "use_reranker": True
            }
        except ValueError:
            # Fallback to default config
            return {
                "intents": [],
                "rag_top_k": 3,
                "use_reranker": True
            }

    async def process_chat_request(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat request using the Orchestrator pattern.

        This follows the Agent Contract:
        - Orchestrator coordinates agent execution
        - Each agent modifies only its assigned fields
        - Shared state flows between agents

        Args:
            request: ChatRequest containing message and metadata

        Returns:
            ChatResponse with generated response

        Raises:
            HTTPException: If processing fails
        """
        if not request.chatbot_id:
            raise HTTPException(
                status_code=400,
                detail="chatbot_id is required"
            )

        # Create LLM client
        llm_client = await self.get_llm_client(
            chatbot_id=request.chatbot_id
        )

        print(f"LLM client created for chatbot_id: {request.chatbot_id}")

        # Create AgentFactory
        agent_factory = await self._create_agent_factory(llm_client)

        # Create ToolRegistry
        tool_registry = await self._create_tool_registry()

        # Create all agents
        agents = {}

        # Create IntentAgent (no tools needed)
        try:
            agents["intent"] = await agent_factory.create_intent_agent(
                tenant_id=request.tenant_id
            )
            print(f"IntentAgent initialized")
        except Exception as e:
            print(f"Warning: Could not initialize IntentAgent: {str(e)}")

        # Create SentimentAgent (no tools needed)
        try:
            agents["sentiment"] = await agent_factory.create_sentiment_agent(
                tenant_id=request.tenant_id
            )
            print(f"SentimentAgent initialized")
        except Exception as e:
            print(f"Warning: Could not initialize SentimentAgent: {str(e)}")

        # Create RAGRetrievalAgent (with FAQTool access)
        try:
            agents["rag"] = agent_factory.create_rag_agent(tool_registry=tool_registry)
            print(f"RAGRetrievalAgent initialized")
        except Exception as e:
            print(f"Warning: Could not initialize RAGRetrievalAgent: {str(e)}")

        # Create PolicyAgent (with ConfigService for policy loading)
        try:
            agents["policy"] = await agent_factory.create_policy_agent(
                tenant_id=request.tenant_id
            )
            print(f"PolicyAgent initialized")
        except Exception as e:
            print(f"Warning: Could not initialize PolicyAgent: {str(e)}")
            # PolicyAgent is optional, so we continue without it

        # Create ResponseAgent (with FAQTool access)
        try:
            agents["response"] = agent_factory.create_response_agent(tool_registry=tool_registry)
            print(f"ResponseAgent initialized")
        except Exception as e:
            print(f"Warning: Could not initialize ResponseAgent: {str(e)}")

        # Get orchestrator configuration
        orchestrator_config = await self._get_orchestrator_config(
            tenant_id=request.tenant_id
        )

        # Create Orchestrator
        orchestrator = Orchestrator(
            agents=agents,
            config=orchestrator_config
        )

        # Process request through orchestrator
        try:
            state = await orchestrator.process_request(request)

            # Log execution summary
            summary = orchestrator.get_execution_summary(state)
            print(f"Execution summary: {summary}")

            # Check for errors
            if state.errors:
                print(f"Errors during processing: {state.errors}")

            # Check if we got a response
            if not state.response:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to generate response"
                )

            # Return chat response
            return ChatResponse(
                response=state.response,
                tenant_id=request.tenant_id,
                chatbot_id=request.chatbot_id,
                session_id=request.session_id
            )

        except Exception as e:
            print(f"Orchestrator processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(e)}"
            )


# Global instance
_chat_service: ChatService = None


async def get_chat_service() -> ChatService:
    """
    Get the global ChatService instance.

    Returns:
        ChatService instance
    """
    global _chat_service

    if _chat_service is None:
        db_manager = await get_db_manager()
        _chat_service = ChatService(db_manager)

    return _chat_service
