"""Generic HTTP API tool for multi-tenant operations."""

import time
import uuid
import httpx
from typing import Optional, Dict, Any
from core.BaseTools import BaseTool
from models.tool import ToolResult, ToolStatus, ToolConfig, AsyncJobInfo


class ApiTool(BaseTool):
    """
    Generic HTTP API caller supporting both sync and async modes.

    Tenant-aware: Uses different endpoints per tenant based on configuration.

    Modes:
    - sync: Executes HTTP request and returns result immediately
    - async: Submits request and returns job_id for polling/webhook
    """

    name = "api_tool"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ApiTool with optional configuration.

        Args:
            config: Dict containing:
                - endpoints: Dict of tenant -> endpoint configs
                - default_headers: Default HTTP headers
                - timeout: Default request timeout
        """
        self._tenant_endpoints: Dict[str, Dict[str, Any]] = config.get("endpoints", {}) if config else {}
        self._default_headers = config.get("default_headers", {}) if config else {}
        self._timeout = config.get("timeout", 30) if config else 30
        self.config = None

    def execute(
        self,
        payload: Dict[str, Any],
        tenant_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Execute API call based on business action and tenant.

        Args:
            payload: Input data (extracted entities, business_action, etc.)
            tenant_id: Tenant identifier
            context: Execution context (session_id, user_id, etc.)

        Returns:
            ToolResult with status, data, and optional async_job
        """
        start_time = time.time()

        # Extract business_action from payload
        business_action = payload.get("business_action")
        if not business_action:
            return ToolResult.failed(
                error="Missing business_action in payload",
                error_code="MISSING_ACTION"
            )

        # Get entities for request body
        entities = payload.get("entities", {})

        # Resolve endpoint configuration
        endpoint_config = self._resolve_endpoint(business_action, tenant_id)
        if not endpoint_config:
            return ToolResult.failed(
                error=f"No endpoint configured for action '{business_action}' (tenant: {tenant_id})",
                error_code="NO_ENDPOINT"
            )

        # Build request
        url = endpoint_config.get("url")
        method = endpoint_config.get("method", "POST").upper()
        execution_mode = endpoint_config.get("execution_mode", "sync")
        headers = {**self._default_headers, **endpoint_config.get("headers", {})}

        # Check if async mode
        if execution_mode == "async":
            return self._execute_async(url, method, entities, headers, business_action, tenant_id, start_time)

        # Sync execution
        return self._execute_sync(url, method, entities, headers, endpoint_config, start_time)

    def _execute_sync(
        self,
        url: str,
        method: str,
        entities: Dict[str, Any],
        headers: Dict[str, str],
        endpoint_config: Dict[str, Any],
        start_time: float
    ) -> ToolResult:
        """Execute synchronous HTTP request."""
        try:
            timeout = endpoint_config.get("timeout", self._timeout)
            retry_attempts = endpoint_config.get("retry_attempts", 0)
            retry_delay = endpoint_config.get("retry_delay_seconds", 1)

            # Build request body (add metadata)
            request_body = {
                "data": entities,
                "metadata": {
                    "timestamp": int(time.time()),
                    "request_id": str(uuid.uuid4())
                }
            }

            # Execute with retries
            response = None
            for attempt in range(retry_attempts + 1):
                try:
                    with httpx.Client(timeout=timeout) as client:
                        if method == "GET":
                            response = client.get(url, params=entities, headers=headers)
                        elif method == "POST":
                            response = client.post(url, json=request_body, headers=headers)
                        elif method == "PUT":
                            response = client.put(url, json=request_body, headers=headers)
                        elif method == "PATCH":
                            response = client.patch(url, json=request_body, headers=headers)
                        elif method == "DELETE":
                            response = client.delete(url, json=request_body, headers=headers)
                        else:
                            return ToolResult.failed(
                                error=f"Unsupported HTTP method: {method}",
                                error_code="INVALID_METHOD"
                            )

                    # Break on success
                    if response.status_code < 500:
                        break

                except httpx.TimeoutException:
                    if attempt < retry_attempts:
                        time.sleep(retry_delay)
                        continue
                    return ToolResult.failed(
                        error=f"Request timeout after {timeout}s",
                        error_code="TIMEOUT"
                    )

            # Process response
            execution_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code >= 400:
                error_msg = f"API returned {response.status_code}"
                try:
                    error_detail = response.json().get("error", response.text)
                    error_msg = f"{error_msg}: {error_detail}"
                except:
                    pass
                return ToolResult.failed(
                    error=error_msg,
                    error_code=f"HTTP_{response.status_code}"
                )

            # Success - parse response
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}

            # Apply response mapping if configured
            if endpoint_config.get("response_mapping"):
                response_data = self._apply_response_mapping(
                    response_data,
                    endpoint_config["response_mapping"]
                )

            return ToolResult.success(
                data=response_data,
                execution_time_ms=execution_time_ms
            )

        except httpx.HTTPError as e:
            return ToolResult.failed(
                error=f"HTTP error: {str(e)}",
                error_code="HTTP_ERROR"
            )
        except Exception as e:
            return ToolResult.failed(
                error=f"Unexpected error: {str(e)}",
                error_code="UNKNOWN_ERROR"
            )

    def _execute_async(
        self,
        url: str,
        method: str,
        entities: Dict[str, Any],
        headers: Dict[str, str],
        business_action: str,
        tenant_id: str,
        start_time: float
    ) -> ToolResult:
        """
        Execute async request (submit job and return job_id).

        In a real implementation, this would:
        1. Submit request to async job queue
        2. Get job_id back
        3. Return pending result with job info

        For now, simulates async behavior.
        """
        job_id = str(uuid.uuid4())

        # In real implementation, would submit to job queue here
        # For now, just return pending status

        async_job = AsyncJobInfo(
            job_id=job_id,
            tool_name=self.name,
            tenant_id=tenant_id,
            poll_url=f"/api/jobs/{job_id}",
            estimated_completion_seconds=30
        )

        return ToolResult.pending(
            async_job=async_job,
            metadata={
                "business_action": business_action,
                "submitted_at": int(start_time)
            }
        )

    def _resolve_endpoint(self, business_action: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Resolve endpoint configuration for action and tenant."""
        # Check tenant-specific endpoint
        if tenant_id in self._tenant_endpoints:
            tenant_actions = self._tenant_endpoints[tenant_id]
            if business_action in tenant_actions:
                return tenant_actions[business_action]

        # Check default endpoint
        if "default" in self._tenant_endpoints:
            default_actions = self._tenant_endpoints["default"]
            if business_action in default_actions:
                return default_actions[business_action]

        return None

    def _apply_response_mapping(self, response_data: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, Any]:
        """Apply field mapping to response data."""
        mapped = {}
        for target_field, source_path in mapping.items():
            parts = source_path.split(".")
            value = response_data
            for part in parts:
                value = value.get(part) if isinstance(value, dict) else None
                if value is None:
                    break
            mapped[target_field] = value

        # Include unmapped fields
        for key, value in response_data.items():
            if key not in mapping.values():
                mapped[key] = value

        return mapped

    def configure(self, config: ToolConfig) -> None:
        """Configure tool with runtime config."""
        self.config = config
        if config.headers:
            self._default_headers = {**self._default_headers, **config.headers}
