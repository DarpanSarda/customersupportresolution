# core/orchestrator.py

from typing import Dict
from models.patch import Patch
from core.StateManager import StateManager
from core.PatchValidator import PatchValidator
from core.BaseAgent import AgentExecutionContext


class Orchestrator:
    """Coordinates agent execution with validation, retry, and state management.

    Orchestrator is the ONLY component that:
        - Instantiates agents
        - Calls agent.execute()
        - Validates patches
        - Applies patches to state
        - Handles retries

    Centralized Control:
        - No agent-to-agent direct calls
        - All state mutations go through StateManager
        - All patches go through PatchValidator
        - Retry policy configured per-agent

    Execution Flow:
        1. Agent lookup (registry)
        2. State preparation (inject input)
        3. Agent instantiation
        4. Agent execution with retry
        5. Patch validation
        6. Patch application
        7. Return updated snapshot
    """

    def __init__(
        self,
        state_manager: StateManager,
        patch_validator: PatchValidator,
        agent_registry: Dict,
        config_loader,
        prompt_loader,
        llm_client,
        tool_registry=None
    ):
        """Initialize orchestrator with required components.

        Args:
            state_manager: Manages state versions and patch application
            patch_validator: Validates patches before application
            agent_registry: Mapping of agent names to agent config with class
                Example: {"IntentAgent": {"class": IntentAgent, "allowed_section": "understanding"}}
            config_loader: Configuration loader
            prompt_loader: Prompt template loader
            llm_client: LLM client instance
            tool_registry: Tool registry for tool execution agents
        """
        self.state_manager = state_manager
        self.patch_validator = patch_validator
        self.agent_registry = agent_registry
        self.config_loader = config_loader
        self.prompt_loader = prompt_loader
        self.llm_client = llm_client
        self.tool_registry = tool_registry

    # -----------------------------------------------------
    # MAIN EXECUTION ENTRYPOINT
    # -----------------------------------------------------

    def execute_agent(
        self,
        agent_name: str,
        request_input: Dict,
        execution_context: AgentExecutionContext,
        session_id: str = "default"
    ) -> Dict:
        """Execute a single agent with full orchestration support.

        This is the primary method for running agents. Handles:
        - Agent lookup and instantiation
        - State preparation (injects user input)
        - Retry policy (configurable per agent)
        - Patch validation
        - State mutation
        - Result return

        Args:
            agent_name: Name of agent to execute (must be in registry)
                Example: "intent-classifier-v1"

            request_input: User request data containing:
                - message: User's input message
                - Other request-specific data

            execution_context: Immutable execution context containing:
                - trace_id: Distributed trace identifier
                - request_id: Unique request identifier
                - tenant_id: Multi-tenant identifier
                - config_version: Active config version
                - prompt_version: Active prompt version
                - otel_span: OpenTelemetry span (optional)
                - langfuse_handler: Langfuse client (optional)
                - logger: Structured logger (optional)

        Returns:
            dict: Updated state snapshot after successful agent execution

        Raises:
            ValueError: If agent_name not in registry
            RuntimeError: If agent fails after all retries

        Process:
            1️⃣ Agent Lookup: Validate agent exists in registry
            2️⃣ Prepare State: Get current state, inject input
            3️⃣ Instantiate Agent: Create agent instance with config
            4️⃣ Execute with Retry: Loop with configurable max_retries
            5️⃣ Validate Patch: Ensure patch complies with all rules
            6️⃣ Apply Patch: Create new state version
            7️⃣ Return Snapshot: Return updated state
        """
        # -------------------------------------------------
        # 1️⃣ Agent Lookup
        # -------------------------------------------------
        if agent_name not in self.agent_registry:
            raise ValueError(f"Agent {agent_name} not registered")

        agent_config = self.agent_registry[agent_name]
        agent_class = agent_config.get("class")

        if agent_class is None:
            raise ValueError(f"Agent {agent_name} has no class defined")

        # -------------------------------------------------
        # 2️⃣ Prepare State (Session-Isolated)
        # -------------------------------------------------
        current_state = self.state_manager.get_session_state(session_id)

        # Inject input into conversation section
        current_state["conversation"]["latest_message"] = request_input.get("message")

        # -------------------------------------------------
        # 3️⃣ Instantiate Agent
        # -------------------------------------------------
        agent_config_data = self.config_loader.get_agent_config(agent_name)
        prompt_version = agent_config_data.get("prompt_version", "v1")

        # Inject dependencies into agent config
        agent_config_data["config_loader"] = self.config_loader
        agent_config_data["prompt_loader"] = self.prompt_loader
        agent_config_data["llm_client"] = self.llm_client
        agent_config_data["tool_registry"] = self.tool_registry

        # Special handling for ToolExecutionAgent (no BaseAgent inheritance)
        if agent_name == "ToolExecutionAgent":
            agent_instance = agent_class(tool_registry=self.tool_registry)
        else:
            agent_instance = agent_class(
                config=agent_config_data,
                prompt=prompt_version
            )

        # -------------------------------------------------
        # 4️⃣ Execute Agent With Retry Policy
        # -------------------------------------------------
        max_retries = agent_config_data.get("max_retries", 0)

        attempt = 0
        last_exception = None

        while attempt <= max_retries:
            try:

                patch: Patch = agent_instance.execute(
                    state=current_state,
                    context=execution_context
                )

                # -------------------------------------------------
                # 5️⃣ Validate Patch
                # -------------------------------------------------
                self.patch_validator.validate(
                    patch=patch,
                    current_state=current_state
                )

                # -------------------------------------------------
                # 6️⃣ Apply Patch (Session-Isolated)
                # -------------------------------------------------
                new_version = self.state_manager.apply_patch(patch, session_id=session_id)

                # -------------------------------------------------
                # 7️⃣ Return Updated Snapshot
                # -------------------------------------------------
                return self.state_manager.get_snapshot(new_version, session_id)

            except Exception as e:
                last_exception = e
                attempt += 1

                if attempt > max_retries:
                    raise RuntimeError(
                        f"Agent {agent_name} failed after retries"
                    ) from last_exception

        # Should never reach here
        raise RuntimeError("Unexpected orchestrator failure")
    
    def execute_system_patch(self, patch: Patch, session_id: str = "default"):
        """Execute a system-generated patch (e.g., lifecycle updates).

        Args:
            patch: Validated patch object
            session_id: Session identifier for isolation

        Returns:
            dict: Updated state snapshot
        """
        current_state = self.state_manager.get_session_state(session_id)
        self.patch_validator.validate(
            patch=patch,
            current_state=current_state
        )
        new_version = self.state_manager.apply_patch(patch, session_id=session_id)
        return self.state_manager.get_snapshot(new_version, session_id)