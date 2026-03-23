"""
ToolRegistry - Manages available tools and agent permissions.

Controls which agents can access which tools.
Integrates with BaseTool interface for unified tool management.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from core.BaseTools import BaseTool
from models.tool import ToolConfig as ToolConfigModel


class ToolConfig(BaseModel):
    """Configuration for a tool."""
    name: str
    description: str
    enabled: bool = True


class ToolPermission(BaseModel):
    """Permission mapping between agents and tools."""
    agent_name: str
    allowed_tools: List[str]


class ToolRegistry:
    """
    Registry for managing tools and agent permissions.

    Controls access: Agent A can only use Tools B, C, etc.
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, Any] = {}  # tool_name -> tool_instance
        self._permissions: Dict[str, List[str]] = {}  # agent_name -> [tool_names]

    def register_tool(self, name: str, tool: Any) -> None:
        """
        Register a tool.

        Args:
            name: Tool name/identifier
            tool: Tool instance (must have execute method)
        """
        self._tools[name] = tool

    def register_tools(self, tools: Dict[str, Any]) -> None:
        """
        Register multiple tools.

        Args:
            tools: Dictionary of tool_name -> tool_instance
        """
        self._tools.update(tools)

    def get_tool(self, name: str) -> Optional[Any]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """
        List all registered tools.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def set_agent_permissions(self, agent_name: str, allowed_tools: List[str]) -> None:
        """
        Set which tools an agent can access.

        Args:
            agent_name: Name of the agent
            allowed_tools: List of tool names this agent can use
        """
        self._permissions[agent_name] = allowed_tools

    def get_agent_tools(self, agent_name: str) -> List[str]:
        """
        Get tools accessible to an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of tool names the agent can use
        """
        return self._permissions.get(agent_name, [])

    def can_use_tool(self, agent_name: str, tool_name: str) -> bool:
        """
        Check if an agent can use a tool.

        Args:
            agent_name: Name of the agent
            tool_name: Name of the tool

        Returns:
            True if agent has permission to use this tool
        """
        allowed_tools = self.get_agent_tools(agent_name)
        return tool_name in allowed_tools

    async def execute_tool(
        self,
        agent_name: str,
        tool_name: str,
        payload: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool on behalf of an agent.

        Args:
            agent_name: Name of the agent requesting execution
            tool_name: Name of the tool to execute
            payload: Input data for the tool

        Returns:
            Result from tool execution

        Raises:
            PermissionError: If agent doesn't have permission
            ValueError: If tool not found
        """
        # Check permissions
        if not self.can_use_tool(agent_name, tool_name):
            raise PermissionError(
                f"Agent '{agent_name}' does not have permission to use tool '{tool_name}'"
            )

        # Get tool
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Execute tool
        if hasattr(tool, "execute"):
            return await tool.execute(payload)
        elif hasattr(tool, "run"):
            return await tool.run(payload)
        elif callable(tool):
            return await tool(payload)
        else:
            raise ValueError(f"Tool '{tool_name}' has no execute method")

    def load_from_config(self, config: Dict[str, Any]) -> None:
        """
        Load tools and permissions from configuration.

        Args:
            config: Configuration dictionary with:
                {
                    "tools": { "tool_name": tool_instance },
                    "permissions": { "agent_name": ["tool1", "tool2"] }
                }
        """
        # Register tools
        if "tools" in config:
            self.register_tools(config["tools"])

        # Set permissions
        if "permissions" in config:
            for agent_name, allowed_tools in config["permissions"].items():
                self.set_agent_permissions(agent_name, allowed_tools)

    def get_permissions_info(self) -> Dict[str, List[str]]:
        """
        Get all agent permissions.

        Returns:
            Dictionary mapping agent names to their allowed tools
        """
        return self._permissions.copy()

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from registry.

        Args:
            tool_name: Name of the tool to remove
        """
        if tool_name in self._tools:
            del self._tools[tool_name]

        # Remove from all permissions
        for agent_name in self._permissions:
            if tool_name in self._permissions[agent_name]:
                self._permissions[agent_name].remove(tool_name)

    def register_base_tool(self, tool: BaseTool) -> None:
        """
        Register a BaseTool instance.

        Args:
            tool: BaseTool instance to register
        """
        tool_name = tool.get_name()
        self._tools[tool_name] = tool

    def register_base_tools(self, tools: List[BaseTool]) -> None:
        """
        Register multiple BaseTool instances.

        Args:
            tools: List of BaseTool instances
        """
        for tool in tools:
            self.register_base_tool(tool)

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool info dict or None if not found
        """
        tool = self.get_tool(tool_name)
        if tool and hasattr(tool, 'get_info'):
            return tool.get_info()
        elif tool:
            return {
                "name": tool_name,
                "type": type(tool).__name__,
                "has_execute": hasattr(tool, 'execute')
            }
        return None

    def list_tools_with_info(self) -> List[Dict[str, Any]]:
        """
        List all registered tools with their info.

        Returns:
            List of tool info dicts
        """
        return [
            self.get_tool_info(tool_name)
            for tool_name in self.list_tools()
            if self.get_tool_info(tool_name) is not None
        ]
