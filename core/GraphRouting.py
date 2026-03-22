"""
Dynamic Graph Routing - Condition-based routing between nodes.

No hardcoded routing logic - fully configurable from database.
"""

from typing import Dict, Any, Optional, List, Callable
from schemas.state import ConversationState


class RoutingCondition:
    """
    Represents a single routing condition.

    Evaluates state and returns True/False.
    """

    def __init__(self, condition_config: Dict[str, Any]):
        """
        Initialize a routing condition from config.

        Args:
            condition_config: Dictionary with condition definition
                Examples:
                    {"field": "detected_intent", "operator": "==", "value": "FAQ_QUERY"}
                    {"field": "should_escalate", "operator": "is_true"}
                    {"field": "sentiment_score", "operator": ">", "value": 0.8}
        """
        self.field = condition_config.get("field")
        self.operator = condition_config.get("operator", "==")
        self.value = condition_config.get("value")

    def evaluate(self, state: ConversationState) -> bool:
        """
        Evaluate the condition against the state.

        Args:
            state: Current conversation state

        Returns:
            True if condition is met, False otherwise
        """
        # Get the field value from state
        if not hasattr(state, self.field):
            return False

        field_value = getattr(state, self.field)

        # Apply operator
        if self.operator == "==":
            return field_value == self.value
        elif self.operator == "!=":
            return field_value != self.value
        elif self.operator == ">":
            return field_value > self.value
        elif self.operator == ">=":
            return field_value >= self.value
        elif self.operator == "<":
            return field_value < self.value
        elif self.operator == "<=":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "is_true":
            return bool(field_value)
        elif self.operator == "is_false":
            return not bool(field_value)
        elif self.operator == "is_null":
            return field_value is None
        elif self.operator == "is_not_null":
            return field_value is not None
        else:
            return False


class RoutingRule:
    """
    Represents a routing rule with conditions and target node.

    If all conditions are met, route to the target node.
    """

    def __init__(
        self,
        target_node: str,
        conditions: List[Dict[str, Any]],
        priority: int = 0
    ):
        """
        Initialize a routing rule.

        Args:
            target_node: Node to route to if conditions are met
            conditions: List of condition configs (all must be true)
            priority: Higher priority rules are checked first
        """
        self.target_node = target_node
        self.conditions = [RoutingCondition(c) for c in conditions]
        self.priority = priority

    def evaluate(self, state: ConversationState) -> bool:
        """
        Evaluate all conditions (AND logic).

        Args:
            state: Current conversation state

        Returns:
            True if all conditions are met
        """
        return all(condition.evaluate(state) for condition in self.conditions)

    def get_target(self) -> str:
        """Get the target node for this rule."""
        return self.target_node


class Router:
    """
    Dynamic router that determines the next node based on state.

    No hardcoded routing logic - loaded from configuration.
    """

    def __init__(self, routing_config: Dict[str, Any]):
        """
        Initialize router from configuration.

        Args:
            routing_config: Dictionary with routing rules
                Format:
                {
                    "node_name": {
                        "default_next": "fallback_node",
                        "rules": [
                            {
                                "target": "target_node",
                                "priority": 1,
                                "conditions": [
                                    {"field": "detected_intent", "operator": "==", "value": "FAQ_QUERY"},
                                    {"field": "intent_meets_threshold", "operator": "is_true"}
                                ]
                            }
                        ]
                    }
                }
        """
        self.routing_rules: Dict[str, List[RoutingRule]] = {}
        self.default_next: Dict[str, str] = {}

        self._parse_routing_config(routing_config)

    def _parse_routing_config(self, config: Dict[str, Any]):
        """
        Parse routing configuration into rules.

        Args:
            config: Routing configuration dictionary
        """
        for node_name, node_config in config.items():
            # Store default next node
            self.default_next[node_name] = node_config.get("default_next")

            # Parse routing rules
            rules = []
            for rule_config in node_config.get("rules", []):
                rule = RoutingRule(
                    target_node=rule_config["target"],
                    conditions=rule_config.get("conditions", []),
                    priority=rule_config.get("priority", 0)
                )
                rules.append(rule)

            # Sort by priority (highest first)
            rules.sort(key=lambda r: r.priority, reverse=True)

            self.routing_rules[node_name] = rules

    def get_next_node(
        self,
        current_node: str,
        state: ConversationState
    ) -> Optional[str]:
        """
        Determine the next node based on current state.

        Args:
            current_node: Current node name
            state: Current conversation state

        Returns:
            Next node name, or None if no matching rule
        """
        # Check routing rules for current node
        rules = self.routing_rules.get(current_node, [])

        for rule in rules:
            if rule.evaluate(state):
                return rule.get_target()

        # Return default next node
        return self.default_next.get(current_node)

    def is_terminal(self, node_name: str) -> bool:
        """
        Check if a node is a terminal (end) node.

        Args:
            node_name: Node name to check

        Returns:
            True if node has no outgoing routes (terminal)
        """
        return (
            node_name not in self.routing_rules and
            node_name not in self.default_next
        )


class RoutingConfigBuilder:
    """
    Builder for creating routing configurations.

    Helper for constructing routing configs from database or code.
    """

    @staticmethod
    def create_simple_route(
        from_node: str,
        to_node: str
    ) -> Dict[str, Any]:
        """
        Create a simple unconditional route.

        Args:
            from_node: Source node
            to_node: Target node

        Returns:
            Routing configuration entry
        """
        return {
            from_node: {
                "default_next": to_node,
                "rules": []
            }
        }

    @staticmethod
    def create_conditional_route(
        from_node: str,
        to_node: str,
        conditions: List[Dict[str, Any]],
        default_next: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a conditional routing rule.

        Args:
            from_node: Source node
            to_node: Target node when conditions are met
            conditions: List of condition dicts
            default_next: Default node if conditions not met

        Returns:
            Routing configuration entry
        """
        return {
            from_node: {
                "default_next": default_next,
                "rules": [
                    {
                        "target": to_node,
                        "priority": 1,
                        "conditions": conditions
                    }
                ]
            }
        }
