"""Observability endpoints for health checks and metrics."""

from fastapi import APIRouter, Response
from services.BootstrapService import bootstrap_system
from utils.Config import CONFIG

router = APIRouter()

# Bootstrap system to get container
container = bootstrap_system(CONFIG)


@router.get("/health")
def health_check():
    """Health check endpoint with observability status."""
    tracing_context = container.tracing_context

    health_status = {
        "status": "healthy",
        "observability": {
            "enabled": tracing_context is not None,
            "langfuse": tracing_context.langfuse is not None if tracing_context else False,
            "otel": tracing_context.otel is not None if tracing_context else False,
            "metrics": tracing_context.metrics is not None if tracing_context else False,
        }
    }

    # Add components status
    health_status["components"] = {
        "graph_engine": container.graph_engine is not None,
        "orchestrator": container.orchestrator is not None,
        "state_manager": container.state_manager is not None,
        "tool_registry": container.tool_registry is not None,
        "logger": container.logger is not None,
    }

    return health_status


@router.get("/metrics")
def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if container.tracing_context and container.tracing_context.metrics:
        from prometheus_client import generate_latest
        return Response(content=generate_latest(), media_type="text/plain")
    return Response(content="# Metrics not enabled\n", media_type="text/plain")
