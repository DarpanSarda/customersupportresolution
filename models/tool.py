"""Tool execution models for async-ready tool layer."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ToolStatus(str, Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"  # For async tools


class AsyncJobInfo(BaseModel):
    """Information about an async job."""
    job_id: str = Field(..., description="Unique job identifier")
    tool_name: str = Field(..., description="Tool that created the job")
    tenant_id: Optional[str] = Field(None, description="Tenant that owns the job")
    status_callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    poll_url: Optional[str] = Field(None, description="URL to poll for job status")
    estimated_completion_seconds: Optional[int] = Field(None, description="Estimated time to completion")


class ToolResult(BaseModel):
    """
    Structured result returned by all tools.

    Supports both sync and async execution modes.
    """
    status: ToolStatus = Field(..., description="Execution status")
    data: Dict[str, Any] = Field(default_factory=dict, description="Result data payload")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    async_job: Optional[AsyncJobInfo] = Field(None, description="Async job info if status is PENDING")

    # Metadata
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @classmethod
    def success(cls, data: Dict[str, Any], metadata: Dict[str, Any] = None, execution_time_ms: int = None) -> "ToolResult":
        """Create a successful result."""
        return cls(
            status=ToolStatus.SUCCESS,
            data=data,
            metadata=metadata or {},
            execution_time_ms=execution_time_ms
        )

    @classmethod
    def failed(cls, error: str, error_code: str = None, metadata: Dict[str, Any] = None) -> "ToolResult":
        """Create a failed result."""
        return cls(
            status=ToolStatus.FAILED,
            error=error,
            error_code=error_code,
            metadata=metadata or {}
        )

    @classmethod
    def pending(cls, async_job: AsyncJobInfo, metadata: Dict[str, Any] = None) -> "ToolResult":
        """Create a pending result for async operations."""
        return cls(
            status=ToolStatus.PENDING,
            async_job=async_job,
            metadata=metadata or {}
        )


class ToolConfig(BaseModel):
    """Configuration for a tool endpoint."""
    url: str = Field(..., description="API endpoint URL")
    method: str = Field("POST", description="HTTP method")
    execution_mode: str = Field("sync", description="sync or async")
    headers: Dict[str, str] = Field(default_factory=dict, description="Static headers")
    auth: Optional[Dict[str, str]] = Field(None, description="Authentication config")
    timeout_seconds: int = Field(30, description="Request timeout")
    retry_attempts: int = Field(0, description="Number of retries on failure")
    retry_delay_seconds: int = Field(1, description="Delay between retries")

    # Response mapping
    response_mapping: Optional[Dict[str, str]] = Field(
        None,
        description="Map response fields to output format"
    )


class ToolCallRecord(BaseModel):
    """Record of a tool execution attempt."""
    tool_name: str = Field(..., description="Name of the tool called")
    status: ToolStatus = Field(..., description="Execution status")
    input_payload: Dict[str, Any] = Field(default_factory=dict, description="Input sent to tool")
    output_payload: Optional[Dict[str, Any]] = Field(None, description="Output received from tool")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    execution_time_ms: Optional[int] = Field(None, description="Execution duration")
    async_job_id: Optional[str] = Field(None, description="Job ID if async")
    tenant_id: Optional[str] = Field(None, description="Tenant who initiated the call")
