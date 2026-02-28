"""Async-aware tool execution agent."""

import time
import json
from datetime import datetime
from models.patch import Patch, PatchMetadata
from models.tool import ToolResult, ToolStatus
from typing import Optional


class ToolExecutionAgent:
    """
    Executes tools and records results in execution section.

    Supports both sync and async tool execution:
    - Sync: Immediate result, status = success/failed
    - Async: Job queued, status = pending, async_job_id tracked
    """

    agent_name = "ToolExecutionAgent"

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def _log_execution(self, context, patch):
        """Log execution like BaseAgent does."""
        print(f"\n{'='*60}")
        print(f"AGENT: {self.agent_name}")
        print(f"{'='*60}")
        print(f"Target Section: {patch.target_section}")
        print(f"Confidence: {patch.confidence}")
        print(f"Changes:")
        print(json.dumps(patch.changes, indent=2))
        print(f"{'='*60}\n")

        if context and context.logger:
            context.logger.info({
                "event": "agent_execution",
                "agent": self.agent_name,
                "trace_id": context.trace_id,
                "request_id": context.request_id,
                "confidence": patch.confidence,
                "target_section": patch.target_section,
            })

    def execute(self, state: dict, context) -> Patch:
        """
        Execute the tool based on decision business action.

        Args:
            state: Current state with decision, policy, and context sections
            context: AgentExecutionContext with trace/request IDs

        Returns:
            Patch with tool execution results (success, failed, or pending)
        """
        patch = self._run(state, context)

        # Log execution
        self._log_execution(context, patch)

        return patch

    def _run(self, state: dict, context) -> Patch:
        """
        Execute the tool based on decision business_action.

        Args:
            state: Current state with decision, policy, and context sections
            context: AgentExecutionContext with trace/request IDs

        Returns:
            Patch with tool execution results (success, failed, or pending)
        """
        decision = state.get("decision", {})
        action = decision.get("action")

        if action != "CALL_TOOL":
            raise ValueError("ToolExecutionAgent called without CALL_TOOL action")

        # Get business action from policy (what to execute)
        policy = state.get("policy", {})
        business_action = policy.get("business_action")

        if not business_action:
            return self._failed_patch(
                error="No business_action in policy section",
                context=context
            )

        # Get tenant_id
        context_data = state.get("context", {})
        tenant_id = context_data.get("tenant_id") or (context.tenant_id if context else "default")

        # Debug logging
        print(f"[DEBUG ToolExecutionAgent] business_action={business_action}, tenant_id={tenant_id}, context_data={context_data}")

        # Get entities from context
        entities = context_data.get("entities", {})

        # Get user's raw input (for tools that need it)
        user_input = state.get("understanding", {}).get("input", {}).get("raw_text", "")

        # Resolve tool for this business action and tenant
        try:
            tool = self.tool_registry.resolve(business_action, tenant_id)
        except ValueError as e:
            return self._failed_patch(error=str(e), context=context)

        # Build payload for tool
        payload = {
            "business_action": business_action,
            "entities": entities,
            "tenant_id": tenant_id,
            "intent": state.get("understanding", {}).get("intent", {}).get("name"),
            "sentiment": state.get("understanding", {}).get("sentiment", {}),
            "user_input": user_input  # Include original user message
        }

        start_time = time.time()
        started_at = datetime.utcnow()

        try:
            # Execute tool (returns ToolResult)
            result: ToolResult = tool.execute(
                payload=payload,
                tenant_id=tenant_id,
                context={"session_id": context.session_id}
            )

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Build tool record based on result status
            if result.status == ToolStatus.SUCCESS:
                tool_record = {
                    "tool_name": tool.name,
                    "status": "success",
                    "input_payload": payload,
                    "output_payload": result.data,
                    "error": None,
                    "started_at": started_at.isoformat(),
                    "completed_at": datetime.utcnow().isoformat(),
                    "execution_time_ms": execution_time_ms or result.execution_time_ms,
                    "tenant_id": tenant_id
                }

                return Patch(
                    agent_name="ToolExecutionAgent",
                    target_section="execution",
                    confidence=1.0,
                    changes={"tools_called": [tool_record]},
                    metadata=self._build_metadata(context, execution_time_ms)
                )

            elif result.status == ToolStatus.PENDING:
                # Async job submitted
                tool_record = {
                    "tool_name": tool.name,
                    "status": "pending",
                    "input_payload": payload,
                    "output_payload": result.data,
                    "error": None,
                    "started_at": started_at.isoformat(),
                    "completed_at": None,
                    "execution_time_ms": execution_time_ms,
                    "async_job_id": result.async_job.job_id if result.async_job else None,
                    "tenant_id": tenant_id
                }

                changes = {
                    "tools_called": [tool_record],
                    "async_jobs_pending": [result.async_job.job_id] if result.async_job else []
                }

                return Patch(
                    agent_name="ToolExecutionAgent",
                    target_section="execution",
                    confidence=1.0,
                    changes=changes,
                    metadata=self._build_metadata(context, execution_time_ms)
                )

            else:  # FAILED
                return self._failed_patch(
                    tool_name=tool.name,
                    error=result.error or "Tool execution failed",
                    error_code=result.error_code,
                    payload=payload,
                    started_at=started_at,
                    context=context,
                    execution_time_ms=execution_time_ms
                )

        except Exception as e:
            # Unexpected errors during tool execution
            execution_time_ms = int((time.time() - start_time) * 1000)
            return self._failed_patch(
                tool_name=tool.name,
                error=f"Unexpected error: {str(e)}",
                payload=payload,
                started_at=started_at,
                context=context,
                execution_time_ms=execution_time_ms
            )

    def _failed_patch(
        self,
        error: str,
        tool_name: Optional[str] = None,
        error_code: Optional[str] = None,
        payload: Optional[dict] = None,
        started_at: Optional[datetime] = None,
        context = None,
        execution_time_ms: int = 0
    ) -> Patch:
        """Create a failed execution patch."""
        tool_record = {
            "tool_name": tool_name or "unknown",
            "status": "failed",
            "input_payload": payload or {},
            "output_payload": None,
            "error": error,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": datetime.utcnow().isoformat(),
            "execution_time_ms": execution_time_ms
        }

        return Patch(
            agent_name="ToolExecutionAgent",
            target_section="execution",
            confidence=1.0,
            changes={"tools_called": [tool_record]},
            metadata=self._build_metadata(context, execution_time_ms)
        )

    def _build_metadata(self, context, execution_time_ms: int = 0) -> PatchMetadata:
        """Build patch metadata."""
        return PatchMetadata(
            execution_time_ms=execution_time_ms,
            config_version=context.config_version if context else None,
            prompt_version=context.prompt_version if context else None,
            trace_id=context.trace_id if context else None,
            request_id=context.request_id if context else None,
            session_id=context.session_id if context else None,
        )
