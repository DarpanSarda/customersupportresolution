"""Tenant-aware tool registry for multi-tenant tool resolution."""

from typing import Optional, Dict, Any
from core.BaseTools import BaseTool
import logging

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Tenant-aware tool registry.

    Resolves tools based on business action and tenant ID.
    Supports tenant-specific tool overrides.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tool registry.

        Args:
            config: Optional configuration dict with:
                - tool_mapping: Business action to tool name mapping
                - tenant_tool_overrides: Tenant-specific tool overrides
        """
        self._tools: Dict[str, BaseTool] = {}
        self._config = config or {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def resolve(
        self,
        business_action: str,
        tenant_id: str = "default"
    ) -> BaseTool:
        """
        Resolve a tool from business action for a tenant.

        Resolution order:
        1. Check tenant-specific tool override
        2. Check global tool mapping
        3. Check if business_action is a direct tool name

        Args:
            business_action: Abstract business action (e.g., "PROCESS_REFUND")
            tenant_id: Tenant identifier

        Returns:
            BaseTool instance

        Raises:
            ValueError: If tool cannot be resolved
        """
        # 1. Check tenant-specific override
        tenant_overrides = self._config.get("tenant_tool_overrides", {})
        if tenant_id in tenant_overrides:
            override_tool = tenant_overrides[tenant_id].get(business_action)
            if override_tool and override_tool in self._tools:
                logger.info(f"Resolved tenant-specific tool: {override_tool} for {business_action} (tenant: {tenant_id})")
                return self._tools[override_tool]

        # 2. Check global tool mapping
        tool_mapping = self._config.get("tool_mapping", {})
        tool_name = tool_mapping.get(business_action)

        if tool_name and tool_name in self._tools:
            logger.info(f"Resolved mapped tool: {tool_name} for {business_action}")
            return self._tools[tool_name]

        # 3. Check if business_action is a direct tool name
        if business_action in self._tools:
            logger.info(f"Resolved direct tool: {business_action}")
            return self._tools[business_action]

        raise ValueError(
            f"Cannot resolve tool for business_action '{business_action}' "
            f"(tenant: {tenant_id}). Available tools: {list(self._tools.keys())}"
        )

    def get(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool

        Returns:
            BaseTool instance

        Raises:
            ValueError: If tool not found
        """
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not registered. Available: {list(self._tools.keys())}")
        return tool

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_tool_config(
        self,
        business_action: str,
        tenant_id: str = "default"
    ) -> Optional[Dict[str, Any]]:
        """
        Get tenant-specific configuration for a business action.

        Args:
            business_action: Business action identifier
            tenant_id: Tenant identifier

        Returns:
            Configuration dict or None
        """
        tenant_configs = self._config.get("tenant_tool_configs", {})
        if tenant_id in tenant_configs:
            return tenant_configs[tenant_id].get(business_action)

        # Return default config if available
        default_configs = self._config.get("default_tool_configs", {})
        return default_configs.get(business_action)

    def configure_tools(self) -> None:
        """
        Apply configuration to all registered tools.

        Tools that support configuration will have their configure() method called.
        """
        for tool_name, tool in self._tools.items():
            # Get tool config (use default as base)
            tool_config = self._config.get("default_tool_configs", {}).get(tool_name)

            if tool_config:
                from models.tool import ToolConfig
                config = ToolConfig(**tool_config)
                tool.configure(config)
                logger.info(f"Configured tool: {tool_name}")
