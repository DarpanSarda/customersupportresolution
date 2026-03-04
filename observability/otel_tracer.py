"""
OpenTelemetry tracer for distributed tracing.

Features:
- Distributed span creation
- Automatic trace propagation
- Agent execution spans
- Tool execution spans
- Exception tracking
- Event logging

Usage:
    tracer = OTELTracer(
        service_name="customer-support-resolution",
        otlp_endpoint="http://localhost:4317"
    )

    with tracer.start_span("agent_execution", attributes={"agent": "IntentAgent"}):
        # Agent logic here
        pass
"""

from typing import Optional, Dict, Any, List, Any as AnyType
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class OTELTracer:
    """
    OpenTelemetry tracer for distributed tracing.

    Provides:
    - Distributed span creation with parent-child relationships
    - Automatic trace ID propagation
    - Exception tracking
    - Event logging within spans
    """

    def __init__(
        self,
        service_name: str = "customer-support-resolution",
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
        enabled: bool = True,
        debug: bool = False
    ):
        """
        Initialize OTEL tracer.

        Args:
            service_name: Service name for resource attribution
            otlp_endpoint: OTLP endpoint (e.g., http://localhost:4317)
            console_export: Enable console export (for debugging)
            sample_rate: Sampling rate (0.0 to 1.0)
            enabled: Enable/disable tracing
            debug: Enable debug logging
        """
        import os

        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4317"
        )
        self.console_export = console_export or os.getenv(
            "OTEL_CONSOLE_EXPORT",
            "false"
        ).lower() == "true"
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.debug = debug
        self.enabled = enabled

        self._tracer_provider: Optional[AnyType] = None
        self._tracer: Optional[AnyType] = None

        if enabled:
            self._initialize()

    def _initialize(self):
        """Initialize OTEL tracer provider."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.trace import Status, StatusCode

            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "1.0.0"
            })

            # Create tracer provider
            provider = TracerProvider(resource=resource)

            # Add console exporter if enabled (for debugging)
            if self.console_export:
                provider.add_span_processor(
                    BatchSpanProcessor(ConsoleSpanExporter())
                )
                if self.debug:
                    logger.info("OTEL console exporter enabled")

            # Add OTLP exporter if endpoint provided
            if self.otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                    otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    if self.debug:
                        logger.info(f"OTEL OTLP exporter configured: {self.otlp_endpoint}")
                except ImportError:
                    logger.warning(
                        "OTLP exporter not available. "
                        "Install: pip install opentelemetry-exporter-otlp-proto-grpc"
                    )

            # Set global provider
            trace.set_tracer_provider(provider)

            # Get tracer
            self._tracer = trace.get_tracer(__name__)
            self._tracer_provider = provider

            if self.debug:
                logger.info("OTEL tracer initialized")

        except ImportError:
            logger.warning(
                "OpenTelemetry packages not installed. "
                "Install: pip install opentelemetry-api opentelemetry-sdk"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"OTEL initialization failed: {e}")
            self.enabled = False

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        parent_context: Optional[AnyType] = None
    ):
        """
        Start a new span.

        Args:
            name: Span name (e.g., "agent.IntentAgent")
            attributes: Span attributes (key-value pairs)
            parent_context: Parent span context (for nesting)

        Yields:
            Span object or None if disabled

        Example:
            with tracer.start_span("agent.IntentAgent", {"tenant_id": "amazon"}) as span:
                # Do work
                span.add_event("processing_complete")
        """
        if not self.enabled or not self._tracer:
            yield None
            return

        # Sample based on rate
        import random
        if random.random() > self.sample_rate:
            yield None
            return

        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        # Use start_as_current_span which properly handles context
        span_builder = self._tracer.start_as_current_span(name)
        # Set attributes after span starts
        if attributes:
            # We need to set attributes on the span, but start_as_current_span doesn't support it directly
            pass

        try:
            with span_builder as span:
                if span and attributes:
                    span.set_attributes(attributes)
                yield span
        except Exception as e:
            # Exception is automatically recorded
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_status(Status(StatusCode.ERROR, str(e)))
            raise

    def add_event(self, name: str, attributes: Dict[str, Any]):
        """
        Add an event to the current span.

        Args:
            name: Event name
            attributes: Event attributes

        Example:
            tracer.add_event("llm_request_sent", {"model": "llama-3.3", "tokens": 100})
        """
        if not self.enabled or not self._tracer:
            return

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.add_event(name, attributes)

    def set_attributes(self, attributes: Dict[str, Any]):
        """
        Set attributes on the current span.

        Args:
            attributes: Attributes to set
        """
        if not self.enabled or not self._tracer:
            return

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attributes(attributes)

    def record_exception(self, exception: Exception, attributes: Optional[Dict[str, Any]] = None):
        """
        Record an exception on the current span.

        Args:
            exception: Exception to record
            attributes: Additional attributes
        """
        if not self.enabled or not self._tracer:
            return

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.record_exception(exception, attributes or {})

    def get_trace_id(self) -> Optional[str]:
        """
        Get current trace ID as hex string.

        Returns:
            Trace ID or None if no active span
        """
        if not self.enabled or not self._tracer:
            return None

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.context:
            return format(current_span.context.trace_id, "032x")
        return None

    def get_span_id(self) -> Optional[str]:
        """
        Get current span ID as hex string.

        Returns:
            Span ID or None if no active span
        """
        if not self.enabled or not self._tracer:
            return None

        from opentelemetry import trace
        current_span = trace.get_current_span()
        if current_span and current_span.context:
            return format(current_span.context.span_id, "016x")
        return None

    def flush(self):
        """Flush all pending spans."""
        if self._tracer_provider:
            try:
                self._tracer_provider.force_flush()
            except Exception as e:
                logger.warning(f"OTEL flush failed: {e}")


class OTELSentry:
    """
    Sentry for manually managing span lifecycle.

    Usage:
        with OTELSentry.start("agent.IntentAgent", tracer, {"tenant": "amazon"}) as sentry:
            # Do work
            sentry.add_event("step_complete")
    """

    @staticmethod
    @contextmanager
    def start(
        name: str,
        tracer: OTELTracer,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Start a span context manager."""
        with tracer.start_span(name, attributes=attributes) as span:
            yield OTELSentry(span, tracer)

    def __init__(self, span, tracer: OTELTracer):
        self._span = span
        self._tracer = tracer

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to this span."""
        if self._span:
            self._span.add_event(name, attributes or {})

    def set_attributes(self, attributes: Dict[str, Any]):
        """Set attributes on this span."""
        if self._span:
            self._span.set_attributes(attributes)

    def set_error(self, exception: Exception):
        """Record error on this span."""
        if self._span:
            self._span.record_exception(exception)


# Span name builders for consistent naming

class SpanNames:
    """Standard span names for different components."""

    @staticmethod
    def agent(agent_name: str) -> str:
        return f"agent.{agent_name}"

    @staticmethod
    def llm(model: str) -> str:
        return f"llm.{model}"

    @staticmethod
    def tool(tool_name: str) -> str:
        return f"tool.{tool_name}"

    @staticmethod
    def vector_search(stage: str) -> str:
        return f"vector_search.{stage}"

    @staticmethod
    def rag_agent() -> str:
        return "agent.RAGAgent"

    @staticmethod
    def graph_execution() -> str:
        return "graph_execution"

    @staticmethod
    def graph(operation: str) -> str:
        return f"graph.{operation}"


class SpanAttributes:
    """Standard span attribute keys."""

    # Agent attributes
    AGENT_NAME = "agent.name"
    AGENT_CONFIDENCE = "agent.confidence"
    AGENT_SECTION = "agent.allowed_section"

    # LLM attributes
    LLM_MODEL = "llm.model"
    LLM_PROVIDER = "llm.provider"
    LLM_PROMPT_TOKENS = "llm.prompt_tokens"
    LLM_COMPLETION_TOKENS = "llm.completion_tokens"
    LLM_TOTAL_TOKENS = "llm.total_tokens"

    # Tool attributes
    TOOL_NAME = "tool.name"
    TOOL_STATUS = "tool.status"
    TOOL_ERROR = "tool.error"

    # Correlation attributes
    TRACE_ID = "trace.id"
    REQUEST_ID = "request.id"
    TENANT_ID = "tenant.id"
    SESSION_ID = "session.id"

    # HTTP attributes
    HTTP_METHOD = "http.method"
    HTTP_ROUTE = "http.route"
    HTTP_STATUS_CODE = "http.status_code"

    @staticmethod
    def create_agent_attributes(
        agent_name: str,
        confidence: Optional[float] = None,
        allowed_section: Optional[str] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standard agent span attributes."""
        attrs = {
            SpanAttributes.AGENT_NAME: agent_name,
        }
        if confidence is not None:
            attrs[SpanAttributes.AGENT_CONFIDENCE] = confidence
        if allowed_section:
            attrs[SpanAttributes.AGENT_SECTION] = allowed_section
        if tenant_id:
            attrs[SpanAttributes.TENANT_ID] = tenant_id
        if session_id:
            attrs[SpanAttributes.SESSION_ID] = session_id
        return attrs

    @staticmethod
    def create_llm_attributes(
        model: str,
        provider: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create standard LLM span attributes."""
        attrs = {
            SpanAttributes.LLM_MODEL: model,
            SpanAttributes.LLM_PROVIDER: provider,
        }
        if prompt_tokens is not None:
            attrs[SpanAttributes.LLM_PROMPT_TOKENS] = prompt_tokens
        if completion_tokens is not None:
            attrs[SpanAttributes.LLM_COMPLETION_TOKENS] = completion_tokens
        if prompt_tokens and completion_tokens:
            attrs[SpanAttributes.LLM_TOTAL_TOKENS] = prompt_tokens + completion_tokens
        return attrs

    @staticmethod
    def create_request_attributes(
        tenant_id: str,
        session_id: str,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create standard request span attributes."""
        attrs = {
            SpanAttributes.TENANT_ID: tenant_id,
            SpanAttributes.SESSION_ID: session_id,
        }
        if request_id:
            attrs[SpanAttributes.REQUEST_ID] = request_id
        return attrs
