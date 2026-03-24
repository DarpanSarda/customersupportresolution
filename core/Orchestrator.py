"""
Orchestrator - Coordinates agent execution and shared state management.

The Orchestrator is responsible for:
1. Creating the shared ConversationState
2. Executing agents in the correct order
3. Applying state updates from each agent
4. Handling errors and retry logic
5. Returning the final state

According to the Agent Contract:
- Agents MUST NOT call other agents directly
- Orchestrator coordinates agent execution
- Each agent only modifies its assigned state fields
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from core.ConversationState import ConversationState, StateUpdate
from schemas.chat import ChatRequest
from models.tool import ToolConfig as ToolConfigModel


class Orchestrator:
    """
    Agent orchestrator for coordinated execution.

    Manages the flow of data between agents by maintaining a shared
    ConversationState that gets updated by each agent in sequence.

    Execution Order:
    1. IntentAgent - Detects user intent
    2. SentimentAgent - Analyzes emotional tone
    3. RAGRetrievalAgent - Retrieves relevant context
    4. PolicyAgent - Evaluates business rules (optional)
    5. ResponseAgent - Generates final response

    Example:
        >>> orchestrator = Orchestrator(agents)
        >>> state = await orchestrator.process_request(chat_request)
        >>> print(state.response)  # Final response
    """

    def __init__(
        self,
        agents: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Orchestrator.

        Args:
            agents: Dict of agent instances by name
                {
                    "intent": IntentAgent,
                    "sentiment": SentimentAgent,
                    "rag": RAGRetrievalAgent,
                    "policy": PolicyAgent,  # Optional
                    "response": ResponseAgent
                }
            config: Optional configuration
        """
        self.agents = agents
        self.config = config or {}

        # Define execution order
        self.execution_order = [
            "intent",          # First: Understand what user wants
            "sentiment",       # Second: Understand emotional tone
            "rag",             # Third: Get relevant context
            "policy",          # Fourth: Evaluate business rules (optional)
            "context_builder", # Fifth: Aggregate all outputs into context bundle
            "response"         # Last: Generate response
        ]

        # Filter out missing optional agents
        self._available_agents = [
            name for name in self.execution_order
            if name in self.agents
        ]

        print(f"🔧 Orchestrator initialized with agents: {list(self.agents.keys())}")
        print(f"📋 Execution order: {self._available_agents}")

    async def process_request(
        self,
        request: ChatRequest,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ConversationState:
        """
        Process a chat request through all agents.

        Args:
            request: ChatRequest with message and metadata
            additional_context: Optional additional context to add to state

        Returns:
            Complete ConversationState with all agent outputs
        """
        start_time = datetime.utcnow()

        # Create initial state from request
        state = ConversationState(
            client_id=request.tenant_id,
            session_id=request.session_id,
            user_message=request.message,
            tenant_id=request.tenant_id
        )

        # Add any additional context
        if additional_context:
            state.context_bundle.update(additional_context)

        # Execute agents in order
        for agent_name in self._available_agents:
            agent = self.agents[agent_name]

            print(f"🔄 Executing {agent_name} agent...")

            try:
                # Execute agent and get state update
                state_update = await self._execute_agent(
                    agent=agent,
                    agent_name=agent_name,
                    state=state,
                    request=request
                )

                # Apply state update
                if state_update:
                    if isinstance(state_update, StateUpdate):
                        state_update.apply(state)
                        print(f"✅ {agent_name} agent completed (StateUpdate)")
                    elif isinstance(state_update, dict):
                        # Legacy dict support - convert to StateUpdate
                        StateUpdate(**state_update).apply(state)
                        print(f"✅ {agent_name} agent completed (dict)")
                    elif hasattr(state_update, "data"):
                        # ResponsePatch support - extract data
                        # Get the actual class name from the agent (not the internal name)
                        actual_agent_name = agent.__class__.__name__
                        self._apply_response_patch(state, actual_agent_name, state_update)
                        print(f"✅ {agent_name} agent completed (ResponsePatch)")
                else:
                    print(f"⚠️  {agent_name} agent returned None")

            except Exception as e:
                # Log error but continue with other agents
                state.add_error(
                    agent_name=agent_name,
                    error=str(e),
                    error_type="agent_execution_error"
                )
                print(f"❌ {agent_name} agent failed: {str(e)}")

                # Check if we should continue or halt
                if self._should_halt_on_error(agent_name):
                    break

        # Calculate processing time
        end_time = datetime.utcnow()
        state.processing_time_ms = int((end_time - start_time).total_seconds() * 1000)

        return state

    async def _execute_agent(
        self,
        agent: Any,
        agent_name: str,
        state: ConversationState,
        request: ChatRequest
    ) -> Optional[StateUpdate]:
        """
        Execute a single agent with the current state.

        Args:
            agent: Agent instance
            agent_name: Name of the agent
            state: Current conversation state
            request: Original chat request

        Returns:
            StateUpdate from agent, or None
        """
        try:
            # Prepare input for agent
            agent_input = self._prepare_agent_input(
                agent_name=agent_name,
                state=state,
                request=request
            )

            # Execute agent
            result = await agent.process(agent_input)

            return result

        except Exception as e:
            # Re-raise to be caught by outer try-except
            raise

    def _prepare_agent_input(
        self,
        agent_name: str,
        state: ConversationState,
        request: ChatRequest
    ) -> Dict[str, Any]:
        """
        Prepare input dict for an agent.

        Each agent gets the state fields it needs.

        Args:
            agent_name: Name of the agent
            state: Current conversation state
            request: Original chat request

        Returns:
            Input dict for the agent
        """
        # Common fields for all agents
        base_input = {
            "message": request.message,
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
            "state": state  # Pass full state for agents that support it
        }

        # Agent-specific inputs
        if agent_name == "intent":
            # IntentAgent needs available intents
            base_input["available_intents"] = self.config.get("intents", [])

        elif agent_name == "sentiment":
            # SentimentAgent needs conversation history (if available)
            base_input["conversation_history"] = state.conversation_history

        elif agent_name == "rag":
            # RAGRetrievalAgent needs query and top_k
            base_input["query"] = request.message
            base_input["top_k"] = self.config.get("rag_top_k", 3)

        elif agent_name == "policy":
            # PolicyAgent needs intent and sentiment
            base_input["intent"] = state.intent
            base_input["sentiment"] = state.sentiment

        elif agent_name == "response":
            # ResponseAgent works with full state
            # (already passed in base_input)
            pass

        return base_input

    def _apply_response_patch(
        self,
        state: ConversationState,
        agent_name: str,
        response_patch: Any
    ):
        """
        Apply ResponsePatch to state (legacy support).

        Maps ResponsePatch data fields to ConversationState fields.
        Handles field name differences between agents and state schema.

        Args:
            state: Conversation state to update
            agent_name: Name of the agent
            response_patch: ResponsePatch object
        """
        if not hasattr(response_patch, "data") or not response_patch.data:
            return

        data = response_patch.data

        # Map ResponsePatch fields to ConversationState fields for each agent
        # This handles field name differences (e.g., "confidence" vs "intent_confidence")

        if agent_name == "IntentAgent":
            # IntentAgent ResponsePatch -> ConversationState
            state.intent = data.get("intent")
            state.intent_confidence = data.get("confidence", 0.0)
            state.intent_raw = {
                "meets_threshold": data.get("meets_threshold"),
                "tool_mapping": data.get("tool_mapping"),
                "reasoning": data.get("reasoning")
            }

        elif agent_name == "SentimentAgent":
            # SentimentAgent ResponsePatch -> ConversationState
            state.sentiment = data.get("sentiment")
            # Map "urgency_score" to "sentiment_confidence" for state
            state.sentiment_confidence = data.get("urgency_score", 0.0)
            state.sentiment_raw = {
                "toxicity_flag": data.get("toxicity_flag"),
                "reasoning": data.get("reasoning"),
                "emotional_indicators": data.get("emotional_indicators"),
                "urgency_score": data.get("urgency_score")
            }

        elif agent_name == "RAGRetrievalAgent":
            # RAGRetrievalAgent ResponsePatch -> ConversationState
            relevant_passages = data.get("relevant_passages", [])
            print(f"[DEBUG] RAGRetrievalAgent: storing {len(relevant_passages)} passages")
            state.rag_results = relevant_passages
            # If we have direct answer (FAQ), use it; otherwise join passages
            source_type = data.get("source_type")
            state.rag_source_type = source_type

            if source_type == "faq":
                # FAQ answer is already formatted
                state.rag_context = data.get("relevant_passages", [""])[0] if data.get("relevant_passages") else ""
            elif source_type == "knowledge_base":
                # Join knowledge base passages
                passages = data.get("relevant_passages", [])
                state.rag_context = "\n\n".join(passages[:3]) if passages else ""
            else:
                state.rag_context = ""

            state.rag_confidence = data.get("confidence_score", 0.0)
            state.rag_raw = data

        elif agent_name == "PolicyAgent":
            # PolicyAgent StateUpdate -> ConversationState
            # PolicyAgent returns StateUpdate with policy_results, policy_action, policy_raw
            state.policy_results = data.get("policy_results", {})
            state.policy_action = data.get("policy_action")
            state.policy_raw = data.get("policy_raw", {})

        elif agent_name == "ResponseAgent":
            # ResponseAgent should return StateUpdate, but handle ResponsePatch for compatibility
            state.response = data.get("response")
            state.response_type = data.get("response_type")

        state._update_timestamp()

    def _should_halt_on_error(self, agent_name: str) -> bool:
        """
        Determine if execution should halt on agent error.

        Args:
            agent_name: Name of the agent that failed

        Returns:
            True if should halt, False to continue
        """
        # Critical agents that should halt execution
        critical_agents = {"response"}  # Must have response

        return agent_name in critical_agents

    def get_execution_summary(self, state: ConversationState) -> Dict[str, Any]:
        """
        Get a summary of the execution.

        Args:
            state: Final conversation state

        Returns:
            Execution summary dict
        """
        return {
            "processing_time_ms": state.processing_time_ms,
            "agents_executed": len(self._available_agents),
            "errors": len(state.errors),
            "warnings": len(state.warnings),
            "intent": state.intent,
            "sentiment": state.sentiment,
            "rag_source": state.rag_source_type,
            "response_type": state.response_type,
            "has_response": state.response is not None
        }


class AgentFactory:
    """
    Factory for creating agent instances and tools.

    This helps the Orchestrator get properly configured agents and tools.
    """

    def __init__(self, llm_client, config_service):
        """
        Initialize AgentFactory.

        Args:
            llm_client: LLM client for agents
            config_service: Config service for agent configuration
        """
        self.llm_client = llm_client
        self.config_service = config_service

    def create_tool_registry(self, tools_config: Optional[Dict[str, Any]] = None):
        """
        Create and configure ToolRegistry with available tools.

        Args:
            tools_config: Optional configuration for tools

        Returns:
            Configured ToolRegistry instance
        """
        from core.ToolRegistry import ToolRegistry
        from tools.FAQTool import FAQTool
        from tools.ApiTool import ApiTool

        # Create tool registry
        registry = ToolRegistry()

        # Register FAQTool
        faq_tool = FAQTool()
        registry.register_base_tool(faq_tool)

        # Register ApiTool
        api_config = tools_config.get("api", {}) if tools_config else {}
        api_tool_config = ToolConfigModel(
            url=api_config.get("url", ""),
            method=api_config.get("method", "POST"),
            timeout_seconds=api_config.get("timeout", 30),
            retry_attempts=api_config.get("retry_attempts", 2),
            headers=api_config.get("headers", {})
        )
        api_tool = ApiTool(config=api_tool_config)
        registry.register_base_tool(api_tool)

        # Set up agent permissions
        # ResponseAgent can use FAQTool
        registry.set_agent_permissions("ResponseAgent", ["FAQTool"])

        # RAGRetrievalAgent can use FAQTool
        registry.set_agent_permissions("RAGRetrievalAgent", ["FAQTool"])

        return registry

    async def create_intent_agent(self, tenant_id: str):
        """Create IntentAgent."""
        from agents.IntentAgent import IntentAgent

        prompt = await self.config_service.get_prompt(
            agent_name="IntentAgent",
            version="v1",
            tenant_id=tenant_id
        )

        return IntentAgent(
            llm_client=self.llm_client,
            system_prompt=prompt
        )

    async def create_sentiment_agent(self, tenant_id: str):
        """Create SentimentAgent."""
        from agents.SentimentAgent import SentimentAgent

        prompt = await self.config_service.get_prompt(
            agent_name="SentimentAgent",
            version="v1",
            tenant_id=tenant_id
        )

        return SentimentAgent(
            llm_client=self.llm_client,
            system_prompt=prompt
        )

    def create_rag_agent(self, tool_registry=None):
        """Create RAGRetrievalAgent."""
        from agents.RAGRetrievalAgent import RAGRetrievalAgent

        print(f"[DEBUG] create_rag_agent called with tool_registry={tool_registry is not None}")
        agent = RAGRetrievalAgent(
            llm_client=self.llm_client,
            tool_registry=tool_registry
        )
        print(f"[DEBUG] RAGRetrievalAgent created, has tool_registry={agent.tool_registry is not None}")
        return agent

    def create_response_agent(self, tool_registry=None):
        """Create ResponseAgent."""
        from agents.ResponseAgent import ResponseAgent

        return ResponseAgent(
            llm_client=self.llm_client,
            tool_registry=tool_registry
        )

    async def create_policy_agent(self, tenant_id: str = "default"):
        """Create PolicyAgent."""
        from agents.PolicyAgent import PolicyAgent

        prompt = await self.config_service.get_prompt(
            agent_name="PolicyAgent",
            version="v1",
            tenant_id=tenant_id
        )

        print(f"[DEBUG] create_policy_agent called with tenant_id={tenant_id}, config_service={self.config_service is not None}")

        agent = PolicyAgent(
            llm_client=self.llm_client,
            system_prompt=prompt,
            config_service=self.config_service
        )

        print(f"[DEBUG] PolicyAgent created, has config_service={agent.config_service is not None}")
        return agent

    def create_context_builder_agent(self, enable_optimization: bool = False):
        """Create ContextBuilderAgent."""
        from agents.ContextBuilderAgent import ContextBuilderAgent

        print(f"[DEBUG] create_context_builder_agent called with enable_optimization={enable_optimization}")

        agent = ContextBuilderAgent(
            llm_client=self.llm_client,
            enable_optimization=enable_optimization
        )

        print(f"[DEBUG] ContextBuilderAgent created")
        return agent
