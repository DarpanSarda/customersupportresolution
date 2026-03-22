"""
Dynamic LangGraph Nodes - Generic node system for any agent.

No hardcoded node types - works with any agent inheriting from BaseAgent.
"""

import asyncio
from typing import Dict, Any, Optional
from schemas.state import ConversationState
from schemas.response import ResponsePatch
from core.BaseAgent import BaseAgent


class GraphNode:
    """
    Generic graph node that works with any BaseAgent.

    No hardcoded types - fully dynamic and configurable.
    """

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        timeout: int = 30,
        is_terminal: bool = False,
        state_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a generic graph node.

        Args:
            name: Node name (e.g., "intent_agent", "sentiment_agent")
            agent: Agent instance (any BaseAgent subclass)
            timeout: Timeout in seconds
            is_terminal: Whether this is an end node
            state_mapping: Optional mapping of agent output to state fields
                         Example: {"intent": "detected_intent", "confidence": "intent_confidence"}
        """
        self.name = name
        self.agent = agent
        self.timeout = timeout
        self.is_terminal = is_terminal
        self.state_mapping = state_mapping or {}

    async def execute(self, state: ConversationState) -> ConversationState:
        """
        Execute the node's agent on the state.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._run_agent(state),
                timeout=self.timeout
            )

            # Update current node in state
            result.current_node = self.name

            return result

        except asyncio.TimeoutError:
            state.error = f"Node '{self.name}' timed out after {self.timeout}s"
            state.retry_count += 1
            return state
        except Exception as e:
            state.error = f"Node '{self.name}' failed: {str(e)}"
            state.retry_count += 1
            return state

    async def _run_agent(self, state: ConversationState) -> ConversationState:
        """
        Run the agent's process method.

        Args:
            state: Current conversation state

        Returns:
            Updated conversation state
        """
        # Initialize agent if needed
        if not self.agent.is_initialized():
            await self.agent.initialize()

        # Build input data for the agent
        input_data = self._build_agent_input(state)

        # Process through agent
        result = await self.agent.process(input_data)

        # Update state with result
        return self._update_state_from_result(state, result)

    def _build_agent_input(self, state: ConversationState) -> Dict[str, Any]:
        """
        Build input data for the agent.

        Each agent receives the full state - they pick what they need.
        Can be overridden by passing state_mapping in constructor.

        Args:
            state: Current conversation state

        Returns:
            Input dictionary for agent.process()
        """
        # Pass the entire state as a dictionary
        # Each agent decides what fields it needs
        return {
            "state": state.dict(),
            "message": state.message,
            "tenant_id": state.tenant_id,
            "session_id": state.session_id
        }

    def _update_state_from_result(
        self,
        state: ConversationState,
        result: ResponsePatch
    ) -> ConversationState:
        """
        Update state with agent result (ResponsePatch).

        Adds the patch to state history and updates state fields based on patch data.

        Args:
            state: Current conversation state
            result: ResponsePatch from agent.process()

        Returns:
            Updated conversation state
        """
        # Add patch to state's patches list
        if "patches" not in state.metadata:
            state.metadata["patches"] = []
        state.metadata["patches"].append(result)

        # Update state fields based on patch type
        if result.data:
            if result.patch_type == "intent":
                state.detected_intent = result.data.get("intent")
                state.intent_confidence = result.data.get("confidence", 0.0)
                state.intent_meets_threshold = result.data.get("meets_threshold", False)

            elif result.patch_type == "sentiment":
                state.detected_sentiment = result.data.get("sentiment")
                state.sentiment_score = result.data.get("score", 0.0)

            elif result.patch_type == "entity":
                state.extracted_entities.update(result.data)

            elif result.patch_type == "response":
                if result.content:
                    state.response = result.content

            elif result.patch_type == "escalation":
                state.should_escalate = True
                state.escalation_reason = result.data.get("reason")
                state.escalation_priority = result.data.get("priority")

        # Track tool usage
        if result.tool_used:
            if "tools_used" not in state.metadata:
                state.metadata["tools_used"] = []
            state.metadata["tools_used"].append(result.tool_used)

        # Apply state_mapping if provided
        if self.state_mapping and result.data:
            for result_key, state_field in self.state_mapping.items():
                if result_key in result.data:
                    setattr(state, state_field, result.data[result_key])

        return state


class NodeFactory:
    """
    Factory for creating nodes from configuration.

    Enables dynamic node creation from database/config.
    """

    @staticmethod
    def create_node(
        name: str,
        agent: BaseAgent,
        config: Optional[Dict[str, Any]] = None
    ) -> GraphNode:
        """
        Create a graph node from configuration.

        Args:
            name: Node name
            agent: Agent instance
            config: Optional configuration dict with:
                - timeout: int
                - is_terminal: bool
                - state_mapping: Dict[str, str]

        Returns:
            Configured GraphNode instance
        """
        config = config or {}

        return GraphNode(
            name=name,
            agent=agent,
            timeout=config.get("timeout", 30),
            is_terminal=config.get("is_terminal", False),
            state_mapping=config.get("state_mapping")
        )

    @staticmethod
    def create_nodes_from_config(
        agents: Dict[str, BaseAgent],
        node_configs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, GraphNode]:
        """
        Create multiple nodes from configuration.

        Args:
            agents: Dictionary of agent name -> agent instance
            node_configs: Dictionary of node name -> node config

        Returns:
            Dictionary of node name -> GraphNode instance
        """
        nodes = {}

        for node_name, node_config in node_configs.items():
            agent_name = node_config.get("agent")
            if agent_name and agent_name in agents:
                nodes[node_name] = NodeFactory.create_node(
                    name=node_name,
                    agent=agents[agent_name],
                    config=node_config
                )

        return nodes
