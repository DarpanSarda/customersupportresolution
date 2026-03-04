"""
Observability package for distributed tracing and metrics.

Provides:
- StructuredLogger: JSON structured logging
- LangfuseTracer: LLM observability (prompts, responses, tokens)
- OTELTracer: Distributed tracing (spans, trace propagation)
- MetricsCollector: Prometheus metrics collection
- TracingContext: Unified context holder

Usage:
    from observability import (
        LangfuseTracer,
        OTELTracer,
        MetricsCollector,
        TracingContext,
        StructuredLogger
    )

    # Initialize tracers
    langfuse = LangfuseTracer()
    otel = OTELTracer()
    metrics = MetricsCollector()

    # Create unified context
    context = TracingContext(
        langfuse=langfuse,
        otel=otel,
        metrics=metrics
    )
"""

from observability.structured_logger import StructuredLogger
from observability.langfuse_tracer import LangfuseTracer, LangfuseSpan, LangfuseObservation, LangfuseTrace
from observability.otel_tracer import OTELTracer, OTELSentry, SpanNames, SpanAttributes
from observability.metrics import MetricsCollector, create_metrics_endpoint
from observability.tracing_context import (
    TracingContext,
    NoOpTracingContext,
    get_tracing_context,
    set_tracing_context,
    reset_tracing_context
)

__all__ = [
    # Structured logging
    "StructuredLogger",

    # Langfuse (LLM tracing)
    "LangfuseTracer",
    "LangfuseSpan",
    "LangfuseObservation",
    "LangfuseTrace",

    # OpenTelemetry (distributed tracing)
    "OTELTracer",
    "OTELSentry",
    "SpanNames",
    "SpanAttributes",

    # Metrics
    "MetricsCollector",
    "create_metrics_endpoint",

    # Unified context
    "TracingContext",
    "NoOpTracingContext",
    "get_tracing_context",
    "set_tracing_context",
    "reset_tracing_context",
]
