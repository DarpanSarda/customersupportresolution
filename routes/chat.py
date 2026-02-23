from fastapi import FastAPI
import uuid

from models.chat import ChatRequest, ChatResponse
from core.BaseAgent import AgentExecutionContext
from bootstrap import bootstrap_system   # your bootstrap function

app = FastAPI()

graph_engine = bootstrap_system()


@app.post("/chat", response_model=ChatResponse)
def chat_route(request: ChatRequest):

    trace_id = str(uuid.uuid4())
    request_id = str(uuid.uuid4())

    execution_context = AgentExecutionContext(
        trace_id=trace_id,
        request_id=request_id,
        tenant_id=request.tenant_id,
        config_version="v1",
        prompt_version="v1"
    )

    result_state = graph_engine.run(
        request_input={"message": request.message},
        execution_context=execution_context
    )

    return ChatResponse(
        session_id=request.session_id,
        state_version=graph_engine.orchestrator.state_manager.current_version,
        response=result_state.get("understanding", {})
    )