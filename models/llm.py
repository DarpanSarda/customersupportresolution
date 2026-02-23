from pydantic import BaseModel
from typing import Optional


class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Optional[dict] = None
    raw: Optional[dict] = None  # For debugging only