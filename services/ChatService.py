"""
ChatService - Service for handling chat requests.

Handles IntentAgent initialization, LLM client creation, and request processing.
"""

from fastapi import HTTPException
from agents.IntentAgent import IntentAgent
from agents.SentimentAgent import SentimentAgent
from llms.BaseLLM import LLMConfig
from llms.LLMFactory import LLMFactory
from utils.DBManager import get_db_manager
from services.ConfigService import ConfigService
from utils.encryption import decrypt
from schemas.chat import ChatRequest, ChatResponse


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

    async def get_intent_agent(
        self,
        llm_client,
        tenant_id: str
    ) -> IntentAgent:
        """
        Create and initialize IntentAgent.

        Args:
            llm_client: LLM client instance
            tenant_id: Tenant identifier

        Returns:
            Initialized IntentAgent

        Raises:
            HTTPException: If prompt not configured
        """
        # Load prompt for IntentAgent
        try:
            prompt = await self.config_service.get_prompt(
                agent_name="IntentAgent",
                version="v1",
                tenant_id=tenant_id
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"IntentAgent prompt not configured: {str(e)}"
            )

        # Initialize IntentAgent
        return IntentAgent(
            llm_client=llm_client,
            system_prompt=prompt
        )

    async def get_sentiment_agent(
        self,
        llm_client,
        tenant_id: str
    ) -> SentimentAgent:
        """
        Create and initialize SentimentAgent.

        Args:
            llm_client: LLM client instance
            tenant_id: Tenant identifier

        Returns:
            Initialized SentimentAgent

        Raises:
            HTTPException: If prompt not configured
        """
        # Load prompt for SentimentAgent
        try:
            prompt = await self.config_service.get_prompt(
                agent_name="SentimentAgent",
                version="v1",
                tenant_id=tenant_id
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"SentimentAgent prompt not configured: {str(e)}"
            )

        # Initialize SentimentAgent
        return SentimentAgent(
            llm_client=llm_client,
            system_prompt=prompt
        )

    async def get_available_intents(self, tenant_id: str):
        """
        Load available intents for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            List of IntentLabel objects

        Raises:
            HTTPException: If no intents configured
        """
        try:
            return await self.config_service.get_intents(
                tenant_id=tenant_id
            )
        except ValueError as e:
            raise HTTPException(
                status_code=500,
                detail=f"No intents configured: {str(e)}"
            )

    async def process_chat_request(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat request.

        Runs both IntentAgent and SentimentAgent to classify the user's message
        and detect emotional tone.

        Args:
            request: ChatRequest containing message and metadata

        Returns:
            ChatResponse with detected intent and sentiment

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

        # Create IntentAgent
        intent_agent = await self.get_intent_agent(
            llm_client=llm_client,
            tenant_id=request.tenant_id
        )

        print(f"IntentAgent initialized for tenant_id: {request.tenant_id}")

        # Create SentimentAgent
        sentiment_agent = await self.get_sentiment_agent(
            llm_client=llm_client,
            tenant_id=request.tenant_id
        )

        print(f"SentimentAgent initialized for tenant_id: {request.tenant_id}")

        # Load available intents
        available_intents = await self.get_available_intents(
            tenant_id=request.tenant_id
        )

        print(f"Available intents loaded: {[intent.label for intent in available_intents]}")

        # Run IntentAgent
        intent_result = await intent_agent.process({
            "message": request.message,
            "available_intents": available_intents,
            "session_id": request.session_id,
            "tenant_id": request.tenant_id
        })

        print(f"IntentAgent result: {intent_result}")

        # Check for intent errors
        if intent_result.data and "error" in intent_result.data:
            raise HTTPException(
                status_code=500,
                detail=intent_result.data["error"]
            )

        # Run SentimentAgent
        sentiment_result = await sentiment_agent.process({
            "message": request.message,
            "conversation_history": [],  # TODO: Load from session if available
            "session_id": request.session_id,
            "tenant_id": request.tenant_id
        })

        print(f"SentimentAgent result: {sentiment_result}")

        # Check for sentiment errors
        if sentiment_result.data and "error" in sentiment_result.data:
            raise HTTPException(
                status_code=500,
                detail=sentiment_result.data["error"]
            )

        # Build response with both intent and sentiment
        detected_intent = intent_result.data.get("intent") if intent_result.data else None
        intent_confidence = intent_result.confidence

        sentiment = sentiment_result.data.get("sentiment") if sentiment_result.data else "neutral"
        urgency_score = sentiment_result.data.get("urgency_score", 0.0) if sentiment_result.data else 0.0
        toxicity_flag = sentiment_result.data.get("toxicity_flag", False) if sentiment_result.data else False

        print(f"Detected intent: {detected_intent} (confidence: {intent_confidence:.2f})")
        print(f"Detected sentiment: {sentiment} (urgency: {urgency_score:.2f}, toxic: {toxicity_flag})")

        # Build response text
        response_parts = [
            f"Detected intent: {detected_intent} (confidence: {intent_confidence:.2f})",
            f"Sentiment: {sentiment} (urgency: {urgency_score:.2f})"
        ]

        if toxicity_flag:
            response_parts.append("⚠️ Toxic language detected - consider escalation")

        if urgency_score >= 0.7:
            response_parts.append("⚠️ High urgency - requires prompt attention")

        response_text = " | ".join(response_parts)

        return ChatResponse(
            response=response_text,
            tenant_id=request.tenant_id,
            chatbot_id=request.chatbot_id,
            session_id=request.session_id
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
