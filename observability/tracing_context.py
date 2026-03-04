"""
Unified tracing context holder.

Holds references to all tracing components (Langfuse, OTEL, Metrics)
and provides a single entry point for observability operations.
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from observability.langfuse_tracer import LangfuseTracer
    from observability.otel_tracer import OTELTracer
    from observability.metrics import MetricsCollector


class TracingContext:
    """
    Unified tracing context for observability.

    Holds references to all tracing components and provides
    convenience methods for common operations.

    Usage:
        context = TracingContext(
            langfuse=langfuse_tracer,
            otel=otel_tracer,
            metrics=metrics_collector
        )

        # Check if tracing is enabled
        if context.enabled:
            context.create_span("agent_execution", ...)
    """

    def __init__(
        self,
        langfuse: Optional["LangfuseTracer"] = None,
        otel: Optional["OTELTracer"] = None,
        metrics: Optional["MetricsCollector"] = None
    ):
        self._langfuse = langfuse
        self._otel = otel
        self._metrics = metrics

    @property
    def langfuse(self) -> Optional["LangfuseTracer"]:
        """Langfuse tracer for LLM observability."""
        return self._langfuse

    @property
    def otel(self) -> Optional["OTELTracer"]:
        """OpenTelemetry tracer for distributed tracing."""
        return self._otel

    @property
    def metrics(self) -> Optional["MetricsCollector"]:
        """Prometheus metrics collector."""
        return self._metrics

    @property
    def enabled(self) -> bool:
        """Check if any tracing is enabled."""
        return bool(
            self._langfuse or
            self._otel or
            self._metrics
        )

    @property
    def has_langfuse(self) -> bool:
        """Check if Langfuse tracing is enabled."""
        return self._langfuse is not None and self._langfuse.enabled

    @property
    def has_otel(self) -> bool:
        """Check if OpenTelemetry tracing is enabled."""
        return self._otel is not None and self._otel.enabled

    @property
    def has_metrics(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._metrics is not None and self._metrics.enabled


class NoOpTracingContext(TracingContext):
    """
    No-op tracing context for testing/disabled observability.

    All operations are safely ignored without errors.
    """

    def __init__(self):
        super().__init__(langfuse=None, otel=None, metrics=None)

    @property
    def enabled(self) -> bool:
        return False

    @property
    def has_langfuse(self) -> bool:
        return False

    @property
    def has_otel(self) -> bool:
        return False

    @property
    def has_metrics(self) -> bool:
        return False


# Global context holder (can be set during bootstrap)
_global_context: Optional[TracingContext] = None


def get_tracing_context() -> TracingContext:
    """Get the global tracing context."""
    global _global_context
    if _global_context is None:
        _global_context = NoOpTracingContext()
    return _global_context


def set_tracing_context(context: TracingContext) -> None:
    """Set the global tracing context."""
    global _global_context
    _global_context = context


def reset_tracing_context() -> None:
    """Reset the global tracing context to no-op."""
    global _global_context
    _global_context = NoOpTracingContext()
