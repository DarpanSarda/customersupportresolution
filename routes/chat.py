from fastapi import APIRouter
import uuid
import time

from models.chat import ChatRequest, ChatResponse
from core.BaseAgent import AgentExecutionContext
from services.BootstrapService import bootstrap_system
from utils.Config import CONFIG
from observability.otel_tracer import SpanNames, SpanAttributes

router = APIRouter()
container = bootstrap_system(CONFIG)
graph_engine = container.graph_engine


@router.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):
    """Handle chat requests with full observability tracing."""

    # Generate correlation IDs
    trace_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    # Get tracing context from container
    tracing_context = container.tracing_context

    # Create root Langfuse trace
    langfuse_trace = None
    if tracing_context and tracing_context.langfuse:
        langfuse_trace = tracing_context.langfuse.create_trace(
            trace_id=trace_id,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            input_message=request.message
        )

    # Create root OTEL span
    otel = tracing_context.otel if tracing_context else None
    root_span = None
    if otel and otel.enabled:
        root_span = otel.start_span(
            name=SpanNames.graph("chat_request"),
            attributes=SpanAttributes.create_request_attributes(
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                request_id=request_id
            )
        )
    else:
        from contextlib import nullcontext
        root_span = nullcontext()

    with root_span:
        start_time = time.time()

        # Record request metric
        metrics = tracing_context.metrics if tracing_context else None
        if metrics and metrics.enabled:
            metrics.record_request(
                tenant_id=request.tenant_id,
                session_id=request.session_id
            )

        try:
            # Create execution context with tracing
            execution_context = AgentExecutionContext(
                trace_id=trace_id,
                request_id=request_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                config_version="v1",
                prompt_version="v1",
                logger=container.logger,
                tracing_context=tracing_context
            )

            # Run graph execution
            result_state = graph_engine.run(
                request_input={"message": request.message},
                execution_context=execution_context,
                session_id=request.session_id
            )

            # Extract final_response from execution section
            execution_section = result_state.get("execution", {})
            final_response = execution_section.get("final_response", "I apologize, but I couldn't generate a response.")

            # Get session-specific version
            state_version = graph_engine.orchestrator.state_manager.get_session_version(request.session_id)

            # Calculate latency
            latency_seconds = time.time() - start_time

            # Finalize Langfuse trace
            if langfuse_trace:
                langfuse_trace.finalize(
                    output=final_response,
                    metadata={
                        "state_version": state_version,
                        "tenant_id": request.tenant_id,
                        "session_id": request.session_id
                    }
                )

            # Record success metric
            if metrics and metrics.enabled:
                metrics.record_request_completion(
                    tenant_id=request.tenant_id,
                    session_id=request.session_id,
                    status="success",
                    latency_seconds=latency_seconds
                )

            # Add OTEL event
            if otel and otel.enabled:
                otel.add_event(
                    "chat_request_success",
                    {
                        "session_id": request.session_id,
                        "state_version": state_version,
                        "response_length": len(final_response)
                    }
                )

            return ChatResponse(
                session_id=request.session_id,
                state_version=state_version,
                response=final_response
            )

        except Exception as e:
            latency_seconds = time.time() - start_time

            # Log error
            if container.logger:
                container.logger.error(
                    {"event": "chat_request_failed", "message": request.message},
                    exception=e,
                    trace_id=trace_id,
                    request_id=request_id
                )

            # Finalize Langfuse trace with error
            if langfuse_trace:
                langfuse_trace.finalize_error(
                    error_message=str(e),
                    error_type=type(e).__name__
                )

            # Record error metric
            if metrics and metrics.enabled:
                metrics.record_request_completion(
                    tenant_id=request.tenant_id,
                    session_id=request.session_id,
                    status="error",
                    latency_seconds=latency_seconds
                )

            # Add OTEL event
            if otel and otel.enabled:
                otel.add_event(
                    "chat_request_error",
                    {
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )

            # Deterministic failure - log error internally in production
            state_version = graph_engine.orchestrator.state_manager.get_session_version(request.session_id)

            return ChatResponse(
                session_id=request.session_id,
                state_version=state_version,
                response="I apologize, but an internal error occurred. Please try again."
            )
