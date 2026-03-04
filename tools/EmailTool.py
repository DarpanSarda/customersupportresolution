"""Email sending tool for notifications and escalations."""

import httpx
from typing import Optional, Dict, Any
from core.BaseTools import BaseTool
from models.tool import ToolResult


class EmailTool(BaseTool):
    """
    Send emails via external email service API.

    Multi-tenant: Each tenant can have their own email service configuration.
    """

    name = "email_tool"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize email tool.

        Args:
            config: Dict with 'endpoints' mapping tenant_id to email service config
                {
                    "endpoints": {
                        "default": {"url": "https://api.emailservice.com/send", "api_key": "..."},
                        "amazon": {"url": "https://ses.amazonaws.com/send", "api_key": "..."}
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
        Send an email.

        Args:
            payload: Contains email details (to, subject, body, from, etc.)
            tenant_id: Tenant identifier
            context: Execution context

        Returns:
            ToolResult with message_id and status
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

        # Build email payload
        email_payload = {
            "to": payload.get("to"),
            "subject": payload.get("subject"),
            "body": payload.get("body"),
            "from": payload.get("from", endpoint.get("default_from", "noreply@example.com")),
            "cc": payload.get("cc"),
            "bcc": payload.get("bcc")
        }

        # Validate required fields
        if not email_payload["to"]:
            return ToolResult.failed(
                error="Missing required field: to",
                error_code="MISSING_TO"
            )

        if not email_payload["subject"]:
            return ToolResult.failed(
                error="Missing required field: subject",
                error_code="MISSING_SUBJECT"
            )

        if not email_payload["body"]:
            return ToolResult.failed(
                error="Missing required field: body",
                error_code="MISSING_BODY"
            )

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, json=email_payload, headers=headers)

            if response.status_code >= 400:
                return ToolResult.failed(
                    error=f"Email API returned {response.status_code}: {response.text}",
                    error_code=f"HTTP_{response.status_code}"
                )

            data = response.json()
            return ToolResult.success(
                data={
                    "message_id": data.get("message_id") or data.get("id"),
                    "status": data.get("status", "sent"),
                    "to": email_payload["to"]
                }
            )

        except httpx.HTTPError as e:
            return ToolResult.failed(
                error=f"HTTP error: {str(e)}",
                error_code="HTTP_ERROR"
            )
        except Exception as e:
            return ToolResult.failed(
                error=f"Failed to send email: {str(e)}",
                error_code="UNKNOWN_ERROR"
            )
