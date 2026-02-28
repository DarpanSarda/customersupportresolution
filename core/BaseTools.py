"""Base tool interface for async-ready tool layer."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from models.tool import ToolResult, ToolConfig


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    All tools must:
    - Be deterministic and stateless
    - Return ToolResult (success, failed, or pending)
    - Support optional configuration
    - Handle tenant-specific execution
    """

    name: str
    config: Optional[ToolConfig] = None

    @abstractmethod
    def execute(
        self,
        payload: Dict[str, Any],
        tenant_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute the tool with the given payload.

        Args:
            payload: Input data for the tool (usually extracted entities)
            tenant_id: Tenant identifier for multi-tenant operations
            context: Additional execution context (session_id, user_id, etc.)

        Returns:
            ToolResult with status, data, and optional async_job info

        Must NOT mutate global state.
        """
        pass

    def configure(self, config: ToolConfig) -> None:
        """Configure the tool with runtime configuration."""
        self.config = config

    def validate_payload(self, payload: Dict[str, Any], required_fields: list) -> tuple[bool, Optional[str]]:
        """
        Validate that payload contains required fields.

        Returns:
            (is_valid, error_message)
        """
        missing = [field for field in required_fields if field not in payload or payload.get(field) is None]
        if missing:
            return False, f"Missing required fields: {', '.join(missing)}"
        return True, None
