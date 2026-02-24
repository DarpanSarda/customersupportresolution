from fastapi import APIRouter
import uuid

from models.chat import ChatRequest, ChatResponse
from core.BaseAgent import AgentExecutionContext
from services.BootstrapService import bootstrap_system
from utils.Config import CONFIG

router = APIRouter()

container = bootstrap_system(CONFIG)
graph_engine = container.graph_engine


@router.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):

    trace_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    execution_context = AgentExecutionContext(
        trace_id=trace_id,
        request_id=request_id,
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        config_version="v1",
        prompt_version="v1"
    )

    try:
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

        return ChatResponse(
            session_id=request.session_id,
            state_version=state_version,
            response=final_response
        )

    except Exception:  # Don't leak stack traces to client
        # Deterministic failure - log error internally in production
        state_version = graph_engine.orchestrator.state_manager.get_session_version(request.session_id)

        return ChatResponse(
            session_id=request.session_id,
            state_version=state_version,
            response="I apologize, but an internal error occurred. Please try again."
        )