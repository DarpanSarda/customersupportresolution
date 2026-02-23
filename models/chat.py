from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    tenant_id: str
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    state_version: int
    response: dict