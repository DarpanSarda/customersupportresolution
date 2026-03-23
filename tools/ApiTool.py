"""
ApiTool - Generic HTTP API caller tool.

Provides ability to make HTTP requests to external APIs with support for:
- Multiple HTTP methods (GET, POST, PUT, PATCH, DELETE)
- Custom headers and authentication
- Request/response transformation
- Retry logic
- Async/sync execution modes
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import httpx
from core.BaseTools import BaseTool
from models.tool import ToolResult, ToolConfig


class ApiTool(BaseTool):
    """
    Generic HTTP API caller tool.

    Supports making HTTP requests to external APIs with configurable
    authentication, headers, retries, and response mapping.

    Example:
        >>> config = ToolConfig(
        ...     url="https://api.example.com/orders/{order_id}",
        ...     method="GET",
        ...     headers={"Authorization": "Bearer {token}"}
        ... )
        >>> tool = ApiTool(config)
        >>> result = await tool.execute({
        ...     "order_id": "12345",
        ...     "token": "abc123",
        ...     "tenant_id": "amazon"
        ... })
    """

    def __init__(self, config: Optional[ToolConfig] = None):
        """
        Initialize ApiTool.

        Args:
            config: Tool configuration including:
                - url: API endpoint URL (supports {variable} placeholders)
                - method: HTTP method (default: POST)
                - headers: Static headers dict
                - auth: Authentication config
                - timeout_seconds: Request timeout (default: 30)
                - retry_attempts: Number of retries (default: 0)
                - retry_delay_seconds: Delay between retries (default: 1)
                - response_mapping: Map response fields to output
        """
        super().__init__(config)

        # HTTP client (lazy loaded)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy load HTTP client."""
        if self._client is None:
            timeout = httpx.Timeout(self.config.timeout_seconds)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute API call.

        Args:
            payload: Input data containing:
                - Path variables for URL placeholders
                - Query parameters
                - Request body (for POST/PUT/PATCH)
                - tenant_id: Tenant identifier (for tenant-specific config)
                - headers_override: Optional headers to override config
                - auth_context: Optional auth context (tokens, API keys)

        Returns:
            ToolResult with API response
        """
        import time
        start_time = time.time()

        try:
            # Build request details
            url = self._build_url(payload)
            method = self.config.method.upper()
            headers = self._build_headers(payload)
            body = self._build_body(payload)
            params = payload.get("params", {})

            # Execute with retry logic
            response = await self._execute_with_retry(
                method=method,
                url=url,
                headers=headers,
                json=body if method in ["POST", "PUT", "PATCH"] else None,
                params=params,
                max_retries=self.config.retry_attempts
            )

            # Parse and map response
            response_data = await self._parse_response(response)
            mapped_data = self._map_response(response_data, payload)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return ToolResult.success(
                data=mapped_data,
                metadata={
                    "status_code": response.status_code,
                    "url": str(response.request.url),
                    "method": method
                },
                execution_time_ms=execution_time_ms
            )

        except httpx.TimeoutException as e:
            return ToolResult.failed(
                error=f"API request timed out: {str(e)}",
                error_code="timeout"
            )

        except httpx.HTTPStatusError as e:
            return ToolResult.failed(
                error=f"HTTP error {e.response.status_code}: {str(e)}",
                error_code=f"http_{e.response.status_code}"
            )

        except httpx.RequestError as e:
            return ToolResult.failed(
                error=f"API request failed: {str(e)}",
                error_code="request_error"
            )

        except Exception as e:
            return ToolResult.failed(
                error=f"Unexpected error: {str(e)}",
                error_code="unexpected_error"
            )

    def _build_url(self, payload: Dict[str, Any]) -> str:
        """
        Build URL by replacing placeholders in config URL.

        Args:
            payload: Input data with variable values

        Returns:
            Resolved URL string
        """
        url_template = self.config.url

        # Replace {variable} placeholders
        for key, value in payload.items():
            placeholder = f"{{{key}}}"
            if placeholder in url_template:
                url_template = url_template.replace(placeholder, str(value))

        return url_template

    def _build_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """
        Build headers combining config and payload.

        Args:
            payload: Input data with headers_override and auth_context

        Returns:
            Complete headers dict
        """
        headers = self.config.headers.copy() if self.config.headers else {}

        # Apply headers from payload
        headers_override = payload.get("headers_override", {})
        headers.update(headers_override)

        # Apply authentication context
        auth_context = payload.get("auth_context", {})
        if auth_context:
            # Replace {token}, {api_key} etc in header values
            for key, value in list(headers.items()):
                for auth_key, auth_value in auth_context.items():
                    placeholder = f"{{{auth_key}}}"
                    if placeholder in str(value):
                        headers[key] = value.replace(placeholder, str(auth_value))

        return headers

    def _build_body(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build request body from payload.

        Args:
            payload: Input data

        Returns:
            Request body dict
        """
        # If body is explicitly provided, use it
        if "body" in payload:
            return payload["body"]

        # Otherwise, use all payload keys that aren't special
        special_keys = {
            "tenant_id", "headers_override", "auth_context",
            "params", "body", "timeout"
        }
        return {k: v for k, v in payload.items() if k not in special_keys}

    async def _execute_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Optional[Dict[str, Any]],
        params: Dict[str, Any],
        max_retries: int
    ) -> httpx.Response:
        """
        Execute HTTP request with retry logic.

        Args:
            method: HTTP method
            url: Request URL
            headers: Request headers
            json: Request body
            params: Query parameters
            max_retries: Maximum retry attempts

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    params=params
                )

                response.raise_for_status()
                return response

            except httpx.HTTPStatusError as e:
                # Don't retry on 4xx errors (client errors)
                if 400 <= e.response.status_code < 500:
                    raise

                last_error = e

                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)

            except httpx.RequestError as e:
                last_error = e

                if attempt < max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)

        # All retries failed
        raise last_error

    async def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Parse HTTP response.

        Args:
            response: HTTP response object

        Returns:
            Parsed response data
        """
        content_type = response.headers.get("content-type", "")

        if "application/json" in content_type:
            return await response.json()

        # Return text for non-JSON responses
        return {
            "text": await response.text(),
            "status_code": response.status_code
        }

    def _map_response(self, response_data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply response mapping if configured.

        Args:
            response_data: Raw API response data
            payload: Original request payload

        Returns:
            Mapped response data
        """
        if not self.config.response_mapping:
            return response_data

        mapped = {}
        for output_key, input_path in self.config.response_mapping.items():
            # Support dot notation for nested paths
            value = response_data
            for key in input_path.split("."):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break

            mapped[output_key] = value

        return mapped

    async def cleanup(self):
        """Cleanup HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_info(self) -> Dict[str, Any]:
        """Get ApiTool information."""
        info = super().get_info()
        info.update({
            "description": "Generic HTTP API caller",
            "supported_methods": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            "retry_enabled": self.config.retry_attempts > 0,
            "timeout_seconds": self.config.timeout_seconds
        })
        return info


class TenantApiTool(ApiTool):
    """
    Tenant-specific API tool with endpoint configuration lookup.

    Loads API endpoint configuration from tenant settings.
    Useful for multi-tenant systems where each tenant has different API endpoints.
    """

    def __init__(
        self,
        config: Optional[ToolConfig] = None,
        tenant_endpoints: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """
        Initialize TenantApiTool.

        Args:
            config: Default tool configuration
            tenant_endpoints: Dict of tenant_id -> endpoint_config
        """
        super().__init__(config)
        self.tenant_endpoints = tenant_endpoints or {}

    async def execute(self, payload: Dict[str, Any]) -> ToolResult:
        """
        Execute API call with tenant-specific configuration.

        Args:
            payload: Input data with tenant_id

        Returns:
            ToolResult with API response
        """
        tenant_id = payload.get("tenant_id", "default")

        # Get tenant-specific endpoint config
        if tenant_id in self.tenant_endpoints:
            # Override config with tenant-specific settings
            tenant_config = self.tenant_endpoints[tenant_id]

            # Create a new ToolConfig with tenant-specific values
            original_config = self.config
            self.config = ToolConfig(
                url=tenant_config.get("url", original_config.url),
                method=tenant_config.get("method", original_config.method),
                headers=tenant_config.get("headers", original_config.headers),
                timeout_seconds=tenant_config.get("timeout", original_config.timeout_seconds),
                retry_attempts=tenant_config.get("retry_attempts", original_config.retry_attempts),
                retry_delay_seconds=tenant_config.get("retry_delay", original_config.retry_delay_seconds),
                response_mapping=tenant_config.get("response_mapping", original_config.response_mapping)
            )

        # Execute with (potentially overridden) config
        result = await super().execute(payload)

        # Restore original config
        if tenant_id in self.tenant_endpoints:
            self.config = original_config

        return result
