"""
GraphEngine - Orchestrates the agent graph execution.

Manages the flow of conversation state through dynamic nodes and routing.
"""

import asyncio
from typing import Dict, Any, Optional, List
from schemas.state import ConversationState, GraphConfig
from core.GraphNodes import GraphNode, NodeFactory
from core.GraphRouting import Router
from core.BaseAgent import BaseAgent


class GraphEngine:
    """
    Orchestrates execution of the agent graph.

    No hardcoded flow - completely dynamic based on configuration.
    """

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        graph_config: GraphConfig
    ):
        """
        Initialize the graph engine.

        Args:
            agents: Dictionary of agent name -> agent instance
            graph_config: Graph configuration with nodes and routing
        """
        self.agents = agents
        self.graph_config = graph_config
        self.nodes: Dict[str, GraphNode] = {}
        self.router: Optional[Router] = None

        self._initialize()

    def _initialize(self):
        """
        Initialize nodes and router from configuration.

        Builds the graph structure dynamically.
        """
        # Create nodes from configuration
        node_configs = self.graph_config.agent_configs
        self.nodes = NodeFactory.create_nodes_from_config(
            agents=self.agents,
            node_configs=node_configs
        )

        # Initialize router with routing rules
        self.router = Router(self.graph_config.routing_rules)

    async def process(
        self,
        message: str,
        tenant_id: str = "default",
        chatbot_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> ConversationState:
        """
        Process a message through the graph.

        Args:
            message: User's message
            tenant_id: Tenant identifier
            chatbot_id: Optional chatbot identifier
            session_id: Optional session identifier
            **kwargs: Additional parameters

        Returns:
            Final conversation state with response
        """
        # Create initial state
        state = ConversationState(
            message=message,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            session_id=session_id,
            current_node=self.graph_config.entry_node
        )

        # Add any additional metadata
        state.metadata.update(kwargs)

        # Execute the graph
        state = await self._execute_graph(state)

        return state

    async def _execute_graph(self, state: ConversationState) -> ConversationState:
        """
        Execute the graph flow from current state.

        Args:
            state: Current conversation state

        Returns:
            Final conversation state
        """
        max_steps = 50  # Prevent infinite loops
        step_count = 0

        while step_count < max_steps:
            current_node = state.current_node

            # Check if we're at a terminal node
            if self.router.is_terminal(current_node):
                break

            # Check if we have an error with too many retries
            if state.error and state.retry_count >= self.graph_config.max_retries:
                # Route to fallback
                state = await self._route_to_fallback(state)
                break

            # Get the node
            node = self.nodes.get(current_node)
            if not node:
                state.error = f"Node '{current_node}' not found"
                break

            # Execute the node
            state = await node.execute(state)

            # Add to history
            state.history.append({
                "node": current_node,
                "result": state.metadata.get(f"{current_node}_result")
            })

            # Determine next node
            next_node = self.router.get_next_node(current_node, state)

            if next_node is None:
                # No route found, we're done
                break

            state.next_node = next_node

            # Check if next node is terminal
            if next_node in self.graph_config.terminal_nodes:
                state.current_node = next_node
                # Execute the terminal node
                terminal_node = self.nodes.get(next_node)
                if terminal_node:
                    state = await terminal_node.execute(state)
                break

            # Move to next node
            state.current_node = next_node

            step_count += 1

        return state

    async def _route_to_fallback(self, state: ConversationState) -> ConversationState:
        """
        Route to fallback node on error.

        Args:
            state: Current conversation state

        Returns:
            Updated state after fallback execution
        """
        fallback_node = self.nodes.get("fallback_agent")
        if fallback_node:
            return await fallback_node.execute(state)

        state.response = "I apologize, but I'm unable to process your request at this time."
        return state

    def get_graph_info(self) -> Dict[str, Any]:
        """
        Get information about the graph.

        Returns:
            Dictionary with graph structure info
        """
        return {
            "graph_name": self.graph_config.graph_name,
            "entry_node": self.graph_config.entry_node,
            "nodes": list(self.nodes.keys()),
            "terminal_nodes": self.graph_config.terminal_nodes,
            "routing_rules": self.graph_config.routing_rules
        }


class GraphEngineBuilder:
    """
    Builder for creating a GraphEngine from configuration.

    Helps construct the graph from database or config files.
    """

    def __init__(self):
        """Initialize the builder."""
        self.agents: Dict[str, BaseAgent] = {}
        self.graph_config = GraphConfig()

    def add_agent(self, name: str, agent: BaseAgent) -> 'GraphEngineBuilder':
        """
        Add an agent to the graph.

        Args:
            name: Agent identifier
            agent: Agent instance

        Returns:
            Self for chaining
        """
        self.agents[name] = agent
        return self

    def set_entry_node(self, node_name: str) -> 'GraphEngineBuilder':
        """
        Set the entry node.

        Args:
            node_name: Starting node name

        Returns:
            Self for chaining
        """
        self.graph_config.entry_node = node_name
        return self

    def add_node_config(
        self,
        node_name: str,
        agent_name: str,
        is_terminal: bool = False,
        timeout: int = 30,
        state_mapping: Optional[Dict[str, str]] = None
    ) -> 'GraphEngineBuilder':
        """
        Add a node configuration.

        Args:
            node_name: Node identifier
            agent_name: Which agent to use
            is_terminal: Whether this is an end node
            timeout: Node timeout in seconds
            state_mapping: Optional output field mapping

        Returns:
            Self for chaining
        """
        self.graph_config.agent_configs[node_name] = {
            "agent": agent_name,
            "is_terminal": is_terminal,
            "timeout": timeout,
            "state_mapping": state_mapping
        }

        if is_terminal and node_name not in self.graph_config.terminal_nodes:
            self.graph_config.terminal_nodes.append(node_name)

        return self

    def add_routing_rule(
        self,
        from_node: str,
        to_node: str,
        conditions: List[Dict[str, Any]],
        priority: int = 0,
        default_next: Optional[str] = None
    ) -> 'GraphEngineBuilder':
        """
        Add a routing rule.

        Args:
            from_node: Source node
            to_node: Target node
            conditions: List of condition dicts
            priority: Rule priority (higher checked first)
            default_next: Default next node if conditions not met

        Returns:
            Self for chaining
        """
        if from_node not in self.graph_config.routing_rules:
            self.graph_config.routing_rules[from_node] = {
                "default_next": default_next,
                "rules": []
            }

        self.graph_config.routing_rules[from_node]["rules"].append({
            "target": to_node,
            "priority": priority,
            "conditions": conditions
        })

        if default_next:
            self.graph_config.routing_rules[from_node]["default_next"] = default_next

        return self

    def build(self) -> GraphEngine:
        """
        Build the GraphEngine.

        Returns:
            Configured GraphEngine instance
        """
        return GraphEngine(
            agents=self.agents,
            graph_config=self.graph_config
        )
