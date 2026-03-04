# core/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models.patch import Patch
import time


class AgentExecutionContext:
    """
    Immutable execution context passed by Orchestrator.

    Contains correlation IDs and tracing components for observability.
    """

    def __init__(
        self,
        trace_id: str,
        request_id: str,
        tenant_id: str,
        config_version: str,
        prompt_version: str,
        session_id: str = "default",
        logger=None,
        tracing_context=None,
    ):
        self.trace_id = trace_id
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.config_version = config_version
        self.prompt_version = prompt_version
        self.session_id = session_id
        self.logger = logger
        self.tracing_context = tracing_context

    @property
    def langfuse(self):
        """Get Langfuse tracer from tracing context."""
        if self.tracing_context:
            return self.tracing_context.langfuse
        return None

    @property
    def otel(self):
        """Get OTEL tracer from tracing context."""
        if self.tracing_context:
            return self.tracing_context.otel
        return None

    @property
    def metrics(self):
        """Get metrics collector from tracing context."""
        if self.tracing_context:
            return self.tracing_context.metrics
        return None


class BaseAgent(ABC):
    """
    Enterprise-grade BaseAgent.

    Guarantees:
    - Stateless execution
    - Patch-only return
    - Section ownership enforcement
    - Confidence validation
    - Prompt + Config version tracking
    """

    agent_name: str
    allowed_section: str

    def __init__(self, config: Dict[str, Any], prompt: str):
        self._config = config
        self._prompt = prompt
        self._tool_registry = config.get("tool_registry")

    # -----------------------------------------------------
    # PUBLIC EXECUTION ENTRYPOINT
    # -----------------------------------------------------

    def execute(
        self,
        state: Dict[str, Any],
        context: AgentExecutionContext,
    ) -> Patch:
        """
        Wrapper enforcing all enterprise constraints.

        Includes:
        - State validation
        - OTEL span creation
        - Metrics recording
        - Patch validation
        - Structured logging
        """

        self._validate_statelessness(state)

        # Start OTEL span if available
        otel = context.otel
        span = None
        if otel and otel.enabled:
            from observability.otel_tracer import SpanNames, SpanAttributes
            span = otel.start_span(
                name=SpanNames.agent(self.agent_name),
                attributes=SpanAttributes.create_agent_attributes(
                    agent_name=self.agent_name,
                    tenant_id=context.tenant_id,
                    session_id=context.session_id
                )
            )
        else:
            from contextlib import nullcontext
            span = nullcontext()

        with span:
            start_time = time.time()

            try:
                patch = self._run(state, context)

                if not isinstance(patch, Patch):
                    raise TypeError(
                        f"{self.agent_name} must return Patch object"
                    )

                self._validate_patch_integrity(patch)

                execution_time_ms = int((time.time() - start_time) * 1000)
                execution_time_seconds = (time.time() - start_time)

                # Inject mandatory metadata
                from models.patch import PatchMetadata

                metadata = PatchMetadata(
                    execution_time_ms=execution_time_ms,
                    config_version=context.config_version,
                    prompt_version=context.prompt_version,
                    trace_id=context.trace_id,
                    request_id=context.request_id,
                    session_id=context.session_id,
                )
                patch.metadata = metadata

                # Record metrics
                metrics = context.metrics
                if metrics and metrics.enabled:
                    metrics.record_agent_execution(
                        agent_name=self.agent_name,
                        tenant_id=context.tenant_id,
                        status="success",
                        latency_seconds=execution_time_seconds,
                        confidence=patch.confidence
                    )

                # Add OTEL event
                if otel and otel.enabled:
                    otel.add_event(
                        "agent_execution_success",
                        {
                            "confidence": patch.confidence,
                            "target_section": patch.target_section,
                            "execution_time_ms": execution_time_ms
                        }
                    )

                # Structured logging
                self._log_execution(context, patch)

                return patch

            except Exception as e:
                # Record error metrics
                metrics = context.metrics
                if metrics and metrics.enabled:
                    execution_time_seconds = (time.time() - start_time)
                    metrics.record_agent_execution(
                        agent_name=self.agent_name,
                        tenant_id=context.tenant_id,
                        status="error",
                        latency_seconds=execution_time_seconds
                    )

                # Re-raise - OTEL span will record exception
                raise

    # -----------------------------------------------------
    # ABSTRACT BUSINESS LOGIC
    # -----------------------------------------------------

    @abstractmethod
    def _run(
        self,
        state: Dict[str, Any],
        context: AgentExecutionContext,
    ) -> Patch:
        """
        Agent implementation.
        Must:
        - Not mutate state
        - Return valid Patch
        - Respect allowed_section
        """
        pass

    # -----------------------------------------------------
    # VALIDATION LAYER
    # -----------------------------------------------------

    def _validate_patch_integrity(self, patch: Patch):

        if patch.agent_name != self.agent_name:
            raise ValueError(
                f"Agent name mismatch. Expected {self.agent_name}"
            )

        if patch.target_section != self.allowed_section:
            raise ValueError(
                f"{self.agent_name} cannot write to {patch.target_section}"
            )

        if not (0.0 <= patch.confidence <= 1.0):
            raise ValueError(
                "Patch confidence must be between 0 and 1"
            )

    def _validate_statelessness(self, state: Dict[str, Any]):
        """
        Prevent accidental mutation patterns.
        Can be extended with hashing if needed.
        """
        if not isinstance(state, dict):
            raise TypeError("State must be dictionary")

    # -----------------------------------------------------
    # TOOL ACCESS (Optional)
    # -----------------------------------------------------

    def _get_tool(self, tool_name: str, tenant_id: str = "default"):
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve
            tenant_id: Tenant identifier

        Returns:
            BaseTool instance or None if tool_registry not available
        """
        if self._tool_registry is None:
            return None
        return self._tool_registry.get(tool_name)

    def _resolve_tool(self, business_action: str, tenant_id: str = "default"):
        """
        Resolve a tool from business action.

        Args:
            business_action: Business action to resolve
            tenant_id: Tenant identifier

        Returns:
            BaseTool instance or None if tool_registry not available
        """
        if self._tool_registry is None:
            return None
        return self._tool_registry.resolve(business_action, tenant_id)

    def _execute_tool(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        tenant_id: str = "default"
    ):
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool
            payload: Tool input payload
            tenant_id: Tenant identifier

        Returns:
            ToolResult or None if tool not available
        """
        tool = self._get_tool(tool_name, tenant_id)
        if tool is None:
            return None
        return tool.execute(payload, tenant_id)

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    def _log_execution(
        self,
        context: AgentExecutionContext,
        patch: Patch
    ):
        import json

        # Print patch details to console
        print(f"\n{'='*60}")
        print(f"AGENT: {self.agent_name}")
        print(f"{'='*60}")
        print(f"Target Section: {patch.target_section}")
        print(f"Confidence: {patch.confidence}")
        print(f"Changes:")
        print(json.dumps(patch.changes, indent=2))
        print(f"{'='*60}\n")

        if context.logger:
            context.logger.info({
                "event": "agent_execution",
                "agent": self.agent_name,
                "trace_id": context.trace_id,
                "request_id": context.request_id,
                "confidence": patch.confidence,
                "target_section": patch.target_section,
            })
