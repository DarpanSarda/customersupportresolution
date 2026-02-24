# core/graph_engine.py


class GraphEngine:
    """Executes agent graphs with conditional routing and lifecycle tracking.

    GraphEngine runs a sequence of agents based on graph configuration:
        - Starts at entry_node or resumes from lifecycle
        - Executes each agent via Orchestrator
        - Evaluates conditions to determine next agent
        - Tracks execution status in lifecycle section
        - Terminates at terminal_nodes

    Graph Config Structure:
        {
            "entry_node": "agent_name",
            "terminal_nodes": ["agent_a", "agent_b"],
            "edges": [
                {
                    "from": "agent_a",
                    "to": "agent_b",
                    "condition": {
                        "section": "understanding",
                        "field": "intent",
                        "equals": "escalation"
                    }
                }
            ]
        }

    Lifecycle Tracking:
        - status: "active" | "completed"
        - current_node: Currently executing agent
        - Enables resumable executions across requests
    """

    def __init__(self, graph_config: dict, orchestrator):
        """Initialize graph engine with configuration and orchestrator.

        Args:
            graph_config: Graph configuration containing:
                - entry_node: Starting agent name
                - terminal_nodes: List of end node names
                - edges: Conditional transitions between agents

            orchestrator: Orchestrator instance for agent execution
        """
        self.graph_config = graph_config
        self.orchestrator = orchestrator

    def run(self, request_input: dict, execution_context, session_id: str = "default") -> dict:
        """Execute the graph from entry/resume node to terminal node.

        Args:
            request_input: User request data containing:
                - message: User's input message
                - Other request-specific data

            execution_context: Immutable execution context containing:
                - trace_id: Distributed trace identifier
                - request_id: Unique request identifier
                - tenant_id: Multi-tenant identifier
                - config_version: Active config version
                - prompt_version: Active prompt version
                - otel_span: OpenTelemetry span (optional)
                - langfuse_handler: Langfuse client (optional)
                - logger: Structured logger (optional)

            session_id: Session identifier for state isolation (default: "default")

        Returns:
            dict: Final state after reaching terminal node

        Raises:
            RuntimeError: If no valid transition exists from current node

        Process:
            1️⃣ Determine Starting Node: Check lifecycle for resume, else entry
            2️⃣ Graph Execution Loop: Execute agents until terminal
            3️⃣ Lifecycle Updates: Track current node and status via patches
        """
        state = self.orchestrator.state_manager.get_session_state(session_id)

        lifecycle = state.get("lifecycle", {})

        # -------------------------------------------------
        # 1️⃣ Determine Starting Node
        # -------------------------------------------------

        if lifecycle.get("status") == "active":
            current_node = lifecycle.get("current_node")
        else:
            current_node = self.graph_config["entry_node"]
            self._update_lifecycle(current_node, "active")

        # -------------------------------------------------
        # 2️⃣ Graph Execution Loop (DAG)
        # -------------------------------------------------

        while True:

            # Execute agent
            state = self.orchestrator.execute_agent(
                agent_name=current_node,
                request_input=request_input,
                execution_context=execution_context,
                session_id=session_id
            )

            # Terminal check
            if current_node in self.graph_config["terminal_nodes"]:
                self._update_lifecycle(current_node, "completed", session_id)
                # Return updated state with completed status
                return self.orchestrator.state_manager.get_session_state(session_id)

            # Evaluate next node
            next_node = self._evaluate_edges(current_node, state)

            if not next_node:
                raise RuntimeError(
                    f"No valid transition from {current_node}"
                )

            self._update_lifecycle(next_node, "active", session_id)
            current_node = next_node

    # -----------------------------------------------------
    # Lifecycle Update via Patch
    # -----------------------------------------------------

    def _update_lifecycle(self, node: str, status: str, session_id: str = "default") -> None:
        """Update lifecycle section via system patch.

        Args:
            node: Current or next node name
            status: "active" or "completed"
            session_id: Session identifier for isolation

        Note:
            Creates a system patch (agent_name="SystemGraphEngine")
            to track execution state without agent involvement.
        """
        lifecycle_patch = {
            "current_node": node,
            "status": status
        }

        from models.patch import Patch
        from core.BaseAgent import AgentExecutionContext

        # SystemPatch (not agent-generated)
        patch = Patch(
            agent_name="SystemGraphEngine",
            target_section="lifecycle",
            confidence=1.0,
            changes=lifecycle_patch,
            metadata={
                "execution_time_ms": 0,
                "config_version": "system",
                "prompt_version": "system",
                "trace_id": "system",
                "request_id": "system",
                "session_id": session_id
            }
        )

        self.orchestrator.execute_system_patch(patch, session_id)

    # -----------------------------------------------------
    # Edge Evaluation
    # -----------------------------------------------------

    def _evaluate_edges(self, from_node: str, state: dict) -> str | None:
        """Evaluate outgoing edges from a node to find next agent.

        Args:
            from_node: Current node name
            state: Current state after agent execution

        Returns:
            Name of next node if condition matches, None otherwise

        Edge Evaluation:
            - Finds all edges from current node
            - Checks each edge's condition against state
            - Returns first matching edge's target
            - Supports wildcard "*" to match any non-null value
        """
        for edge in self.graph_config["edges"]:

            if edge["from"] != from_node:
                continue

            condition = edge["condition"]

            section = condition["section"]
            field_path = condition["field"]
            expected = condition["equals"]

            actual = self._extract_field(state[section], field_path)

            # Wildcard match: "*" matches any non-null value
            if expected == "*":
                if actual is not None:
                    return edge["to"]
            elif actual == expected:
                return edge["to"]

        return None

    def _extract_field(self, data: dict, field_path: str):
        """Extract nested field value using dot notation with array indexing.

        Args:
            data: Dictionary to extract from (typically a state section)
            field_path: Dot-separated field path with optional array index
                Examples: "intent.primary", "tools_called[-1].status"

        Returns:
            Field value if path exists, None otherwise

        Examples:
            _extract_field({"intent": {"primary": "refund"}}, "intent.primary") -> "refund"
            _extract_field({"tools_called": [{"status": "success"}]}, "tools_called[-1].status") -> "success"
            _extract_field({"tools_called": []}, "tools_called[-1].status") -> None
        """
        # Remove brackets and split
        path = field_path.replace("]", "").replace("[", ".")
        parts = path.split(".")
        value = data

        for part in parts:
            if part == "":
                continue  # Skip empty parts from double dots

            # Handle negative indices ([-1] for last element)
            if value is None:
                return None

            # Check if part is a digit (array index)
            if part.lstrip("-").isdigit():
                index = int(part)
                if isinstance(value, list) and -len(value) <= index < len(value):
                    value = value[index]
                else:
                    return None
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value