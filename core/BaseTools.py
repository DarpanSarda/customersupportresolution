"""
BaseTools - Abstract base class and interfaces for all tools.

All tools (FAQTool, ApiTool, etc.) must inherit from BaseTool.
This ensures unified interface and methods across all tools.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from models.tool import ToolResult, ToolConfig


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Enforces a unified interface for tool initialization and execution.
    All tools must implement the execute method and return a ToolResult.
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize the base tool.

        Args:
            config: Tool configuration (optional)
        """
        self.config = config or ToolConfig(url="", method="GET")
        self.name = self.__class__.__name__

    @abstractmethod
    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given payload.

        This is the main method that each tool must implement.

        Args:
            payload: Input data for the tool

        Returns:
            ToolResult with status, data, error (if failed)
        """
        pass

    def get_name(self) -> str:
        """Get the tool name."""
        return self.name

    def get_config(self) -> ToolConfig:
        """Get the tool configuration."""
        return self.config

    def validate_payload(self, payload: Dict[str, Any], required_fields: list) -> tuple[bool, Optional[str]]:
        """
        Validate that payload contains required fields.

        Args:
            payload: Input payload to validate
            required_fields: List of required field names

        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_fields = [field for field in required_fields if field not in payload]
        if missing_fields:
            return False, f"Missing required fields: {', '.join(missing_fields)}"
        return True, None

    def get_info(self) -> Dict[str, Any]:
        """
        Get tool information.

        Returns:
            Dictionary with tool details
        """
        return {
            "name": self.name,
            "type": "tool",
            "config": self.config.model_dump() if self.config else {}
        }
