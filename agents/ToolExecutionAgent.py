import time
from models.patch import Patch, PatchMetadata

class ToolExecutionAgent:
    """Executes tools and records results in execution section.

    Hardened failure handling:
    - Tool lookup inside try block
    - Strict output type validation
    - No crash propagation
    - Deterministic failure record
    - Always records tool call
    """

    def __init__(self, tool_registry):
        self.tool_registry = tool_registry

    def execute(self, state: dict, context):
        """Execute the tool specified in decision.route.

        Args:
            state: Current state with decision section
            context: AgentExecutionContext with trace/request IDs

        Returns:
            Patch with tool execution results (success or failed)
        """
        decision = state.get("decision", {})
        action = decision.get("action")

        if action != "CALL_TOOL":
            raise ValueError("ToolExecutionAgent called without CALL_TOOL action")

        tool_name = decision.get("route")

        # Build payload from state
        understanding = state.get("understanding", {})
        payload = {
            "user_input": understanding.get("input", {}).get("raw_text", ""),
            "intent": understanding.get("intent", {}).get("name")
        }

        start_time = time.time()

        # Initialize failure state
        status = "failed"
        result = None
        error = None

        try:
            # Tool lookup inside try block (handles not registered)
            tool = self.tool_registry.get(tool_name)

            # Execute tool
            result = tool.execute(payload)

            # Strict output type validation
            if not isinstance(result, dict):
                raise ValueError(f"Tool {tool_name} must return dict, got {type(result).__name__}")

            status = "success"

        except ValueError as e:
            # Validation errors (wrong type, etc.)
            error = str(e)
            result = None
            status = "failed"

        except Exception as e:
            # Tool execution errors
            error = str(e)
            result = None
            status = "failed"

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Always record tool call (deterministic)
        tool_record = {
            "tool_name": tool_name,
            "status": status,
            "input_payload": payload,
            "output_payload": result,
            "error": error
        }

        # Create complete metadata
        metadata = PatchMetadata(
            execution_time_ms=execution_time_ms,
            config_version=context.config_version,
            prompt_version=context.prompt_version,
            trace_id=context.trace_id,
            request_id=context.request_id,
            session_id=context.session_id,
        )

        patch = Patch(
            agent_name="ToolExecutionAgent",
            target_section="execution",
            confidence=1.0,
            changes={
                "tools_called": [tool_record]
            },
            metadata=metadata
        )

        return patch