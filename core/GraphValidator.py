class GraphValidator:
    """Validates LangGraph configuration before execution.

    Ensures graph structure is valid and safe:
        - Entry node exists and is reachable
        - All nodes reference registered agents
        - All edges reference valid nodes
        - Graph is acyclic (DAG enforcement - no circular dependencies)

    Graph Config Structure:
        {
            "nodes": {
                "agent_name": {...agent config...}
            },
            "edges": [
                {"from": "agent_a", "to": "agent_b"}
            ],
            "entry_node": "first_agent"
        }
    """

    def validate(self, graph_config: dict, agent_registry: dict) -> None:
        """Validate graph configuration against agent registry.

        Args:
            graph_config: Graph configuration containing:
                - nodes: Dict of node_name -> node_config
                - edges: List of {"from": node_a, "to": node_b}
                - entry_node: Name of starting node

            agent_registry: Dict of registered agent names

        Raises:
            ValueError: If validation fails with specific reason:
                - "Entry node not defined in nodes": entry_node missing
                - "Node {node} not registered agent": Node not in registry
                - "Invalid edge from {node}": Edge references non-existent node
                - "Invalid edge to {node}": Edge references non-existent node
                - "Cycle detected in graph": Graph has circular dependency

        Validation Checks:
            1️⃣ Entry node exists in nodes
            2️⃣ All nodes reference registered agents
            3️⃣ All edges reference valid nodes
            4️⃣ Graph is acyclic (DAG)
        """
        nodes = graph_config["nodes"].keys()
        edges = graph_config["edges"]
        entry = graph_config["entry_node"]

        # 1️⃣ Entry exists
        if entry not in nodes:
            raise ValueError("Entry node not defined in nodes")

        # 2️⃣ All nodes must exist in agent registry
        for node in nodes:
            if node not in agent_registry:
                raise ValueError(f"Node {node} not registered agent")

        # 3️⃣ Edge validation
        for edge in edges:
            if edge["from"] not in nodes:
                raise ValueError(f"Invalid edge from {edge['from']}")
            if edge["to"] not in nodes:
                raise ValueError(f"Invalid edge to {edge['to']}")

        # 4️⃣ Cycle detection (DAG enforcement)
        self._ensure_acyclic(nodes, edges)

    def _ensure_acyclic(self, nodes, edges) -> None:
        """Detect cycles in graph using DFS with recursion stack.

        Args:
            nodes: Iterable of node names
            edges: List of {"from": node_a, "to": node_b}

        Raises:
            ValueError: If cycle is detected in graph

        Algorithm:
            - Build adjacency list from edges
            - DFS traversal with visited set and recursion stack
            - If node encountered in stack, cycle exists
        """
        adjacency = {node: [] for node in nodes}
        for edge in edges:
            adjacency[edge["from"]].append(edge["to"])

        visited = set()
        stack = set()

        def dfs(node):
            if node in stack:
                raise ValueError("Cycle detected in graph")
            if node in visited:
                return

            stack.add(node)
            for neighbor in adjacency[node]:
                dfs(neighbor)
            stack.remove(node)
            visited.add(node)

        for node in nodes:
            dfs(node)