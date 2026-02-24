from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from uuid import uuid4

class PatchMetadata(BaseModel):
    execution_time_ms: int
    config_version: str
    prompt_version: str
    trace_id: str
    request_id: str

class Patch(BaseModel):
    patch_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_name: str
    target_section: str
    confidence: float
    changes: Dict[str, Any]
    metadata: Optional[PatchMetadata] = None

    class Config:
        extra = "forbid"

    @validator("confidence")
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        return v

    @validator("agent_name")
    def validate_agent_name(cls, v):
        if not v:
            raise ValueError("agent_name cannot be empty")
        return v

    @validator("target_section")
    def validate_target_section(cls, v):
        if not v:
            raise ValueError("target_section cannot be empty")
        return v