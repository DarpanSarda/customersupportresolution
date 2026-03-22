"""
BaseAgent - Abstract base class for all agents.

All agents (IntentAgent, SentimentAgent, PolicyAgent, etc.) must inherit from BaseAgent.
This ensures unified interface and methods across all agents.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from llms.BaseLLM import BaseLLM
from schemas.response import ResponsePatch


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Enforces a unified interface for agent initialization, execution, and lifecycle.
    """

    def __init__(
        self,
        llm_client: BaseLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional['ToolRegistry'] = None
    ):
        """
        Initialize the base agent.

        Args:
            llm_client: LLM client for inference
            system_prompt: System prompt for the agent
            config: Additional agent-specific configuration
            tool_registry: ToolRegistry for tool access (with permissions)
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.config = config or {}
        self.tool_registry = tool_registry
        self._initialized = False

    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> ResponsePatch:
        """
        Process input and return a response patch.

        This is the main method that each agent must implement.
        Each agent returns a ResponsePatch (not a full response).

        Args:
            input_data: Input data to process (message, context, etc.)
            **kwargs: Additional parameters

        Returns:
            ResponsePatch with this agent's contribution
        """
        pass

    async def initialize(self):
        """
        Initialize the agent (lazy initialization).

        Called before first use. Override in subclass if needed.
        """
        if not self._initialized:
            await self._setup()
            self._initialized = True

    async def _setup(self):
        """
        Agent-specific setup logic.

        Override in subclass if initialization is needed.
        """
        pass

    async def cleanup(self):
        """
        Cleanup resources.

        Override in subclass if cleanup is needed.
        """
        pass

    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.

        Returns:
            System prompt string
        """
        return self.system_prompt or self._get_default_system_prompt()

    @abstractmethod
    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt for the agent.

        Each agent must provide its default prompt.

        Returns:
            Default system prompt string
        """
        pass

    def update_config(self, key: str, value: Any):
        """
        Update agent configuration.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config[key] = value

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def is_initialized(self) -> bool:
        """Check if agent is initialized."""
        return self._initialized

    def get_agent_name(self) -> str:
        """Get agent class name."""
        return self.__class__.__name__

    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary with agent details
        """
        return {
            "name": self.get_agent_name(),
            "initialized": self._initialized,
            "config": self.config,
            "has_system_prompt": bool(self.system_prompt),
            "has_tool_registry": self.tool_registry is not None
        }

    # ============ TOOL ACCESS ============

    async def use_tool(
        self,
        tool_name: str,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Use a tool (with permission check).

        Args:
            tool_name: Name of the tool to use
            payload: Input data for the tool

        Returns:
            Result from tool execution

        Raises:
            PermissionError: If agent doesn't have permission
            ValueError: If tool not found or no tool_registry set
        """
        if not self.tool_registry:
            raise ValueError("No tool_registry configured for this agent")

        return await self.tool_registry.execute_tool(
            agent_name=self.get_agent_name(),
            tool_name=tool_name,
            payload=payload
        )

    def can_use_tool(self, tool_name: str) -> bool:
        """
        Check if this agent can use a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            True if agent has permission to use this tool
        """
        if not self.tool_registry:
            return False

        return self.tool_registry.can_use_tool(
            agent_name=self.get_agent_name(),
            tool_name=tool_name
        )

    def get_available_tools(self) -> list:
        """
        Get list of tools this agent can use.

        Returns:
            List of tool names
        """
        if not self.tool_registry:
            return []

        return self.tool_registry.get_agent_tools(self.get_agent_name())
