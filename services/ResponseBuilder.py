"""
ResponseBuilder - Builds final response from agent patches.

Combines patches from all agents into coherent final response.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from schemas.response import ResponsePatch, FinalResponse


class ResponseBuilder:
    """
    Builds the final response from agent patches.

    Combines all patches into a coherent final response.
    """

    def __init__(self):
        """Initialize the response builder."""
        self.patches: List[ResponsePatch] = []
        self.agent_timing: Dict[str, float] = {}

    def add_patch(self, patch: ResponsePatch) -> None:
        """
        Add a patch from an agent.

        Args:
            patch: ResponsePatch to add
        """
        self.patches.append(patch)

    def create_patch(
        self,
        agent_name: str,
        patch_type: str,
        content: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        tool_used: Optional[str] = None,
        tool_result: Optional[Dict[str, Any]] = None
    ) -> ResponsePatch:
        """
        Create a new patch.

        Args:
            agent_name: Name of the agent
            patch_type: Type of patch
            content: Text content
            data: Structured data
            confidence: Confidence score
            tool_used: Tool that was used
            tool_result: Result from tool

        Returns:
            ResponsePatch object
        """
        return ResponsePatch(
            agent_name=agent_name,
            patch_type=patch_type,
            content=content,
            data=data,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_used=tool_used,
            tool_result=tool_result
        )

    def build(
        self,
        response_text: str,
        tenant_id: str,
        session_id: Optional[str] = None,
        chatbot_id: Optional[str] = None,
        processing_time: float = 0.0
    ) -> FinalResponse:
        """
        Build the final response from all patches.

        Args:
            response_text: Final response text
            tenant_id: Tenant identifier
            session_id: Optional session identifier
            chatbot_id: Optional chatbot identifier
            processing_time: Total processing time

        Returns:
            FinalResponse object
        """
        # Extract data from patches
        detected_intent = None
        detected_sentiment = None
        extracted_entities = {}
        tools_used = []
        escalated = False
        escalation_reason = None

        for patch in self.patches:
            # Extract intent
            if patch.patch_type == "intent" and patch.data:
                detected_intent = patch.data.get("intent")

            # Extract sentiment
            if patch.patch_type == "sentiment" and patch.data:
                detected_sentiment = patch.data.get("sentiment")

            # Extract entities
            if patch.patch_type == "entity" and patch.data:
                extracted_entities.update(patch.data)

            # Track tools
            if patch.tool_used:
                tools_used.append(patch.tool_used)

            # Check for escalation
            if patch.patch_type == "escalation" and patch.data:
                escalated = True
                escalation_reason = patch.data.get("reason")

        return FinalResponse(
            response=response_text,
            patches=self.patches,
            detected_intent=detected_intent,
            detected_sentiment=detected_sentiment,
            extracted_entities=extracted_entities,
            tenant_id=tenant_id,
            session_id=session_id,
            chatbot_id=chatbot_id,
            tools_used=tools_used,
            escalated=escalated,
            escalation_reason=escalation_reason,
            total_processing_time=processing_time,
            agent_timing=self.agent_timing
        )

    def get_patches_by_type(self, patch_type: str) -> List[ResponsePatch]:
        """
        Get all patches of a specific type.

        Args:
            patch_type: Type of patches to get

        Returns:
            List of patches of the specified type
        """
        return [p for p in self.patches if p.patch_type == patch_type]

    def get_patches_by_agent(self, agent_name: str) -> List[ResponsePatch]:
        """
        Get all patches from a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of patches from the specified agent
        """
        return [p for p in self.patches if p.agent_name == agent_name]

    def clear(self) -> None:
        """Clear all patches and timing data."""
        self.patches.clear()
        self.agent_timing.clear()
