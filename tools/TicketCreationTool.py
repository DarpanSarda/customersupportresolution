"""Ticket creation tool for support systems."""

import httpx
from typing import Optional, Dict, Any
from core.BaseTools import BaseTool
from models.tool import ToolResult


class TicketCreationTool(BaseTool):
    """
    Create support tickets via external ticket system API.

    Multi-tenant: Each tenant can have their own ticket system endpoint.
    """

    name = "ticket_creation_tool"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ticket creation tool.

        Args:
            config: Dict with 'endpoints' mapping tenant_id to endpoint config
                {
                    "endpoints": {
                        "default": {"url": "https://api.ticketsystem.com/create", "api_key": "..."},
                        "amazon": {"url": "https://amazon.support.com/api/tickets", "api_key": "..."}
                    }
                }
        """
        self._endpoints = config.get("endpoints", {}) if config else {}

    def execute(
        self,
        payload: Dict[str, Any],
        tenant_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Create a support ticket.

        Args:
            payload: Contains ticket details (title, description, priority, customer_info, etc.)
            tenant_id: Tenant identifier
            context: Execution context

        Returns:
            ToolResult with ticket_id and status
        """
        # Get tenant endpoint
        endpoint = self._endpoints.get(tenant_id) or self._endpoints.get("default")
        if not endpoint:
            return ToolResult.failed(
                error=f"No endpoint configured for tenant: {tenant_id}",
                error_code="NO_ENDPOINT"
            )

        url = endpoint.get("url")
        api_key = endpoint.get("api_key")

        if not url:
            return ToolResult.failed(
                error=f"No URL configured for tenant: {tenant_id}",
                error_code="NO_URL"
            )

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Build ticket payload
        ticket_payload = {
            "title": payload.get("title", "Support Ticket"),
            "description": payload.get("description", ""),
            "priority": payload.get("priority", "MEDIUM"),
            "customer_info": payload.get("customer_info", {}),
            "metadata": payload.get("metadata", {})
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, json=ticket_payload, headers=headers)

            if response.status_code >= 400:
                return ToolResult.failed(
                    error=f"Ticket API returned {response.status_code}: {response.text}",
                    error_code=f"HTTP_{response.status_code}"
                )

            data = response.json()
            return ToolResult.success(
                data={
                    "ticket_id": data.get("ticket_id") or data.get("id"),
                    "status": data.get("status", "created"),
                    "url": data.get("url")
                }
            )

        except httpx.HTTPError as e:
            return ToolResult.failed(
                error=f"HTTP error: {str(e)}",
                error_code="HTTP_ERROR"
            )
        except Exception as e:
            return ToolResult.failed(
                error=f"Failed to create ticket: {str(e)}",
                error_code="UNKNOWN_ERROR"
            )
