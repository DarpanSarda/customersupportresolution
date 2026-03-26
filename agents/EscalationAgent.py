"""
EscalationAgent - Determines when and how to escalate conversations.

This agent evaluates conversation state to decide if escalation is needed based on:
- Sentiment analysis (anger, frustration, toxicity)
- Policy evaluation results (blocked actions, required escalation)
- Urgency/confidence thresholds from SentimentAgent and PolicyAgent

According to the Agent Contract:
- MUST only read from shared state (sentiment, sentiment_confidence, policy_results, etc.)
- MUST only return state updates (escalation_triggered, escalation_channel, escalation_details)
- MUST use tools for actual escalation actions (EmailTool, SlackTool, TicketCreationTool, WebhookTool)
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from core.BaseAgent import BaseAgent
from core.ConversationState import ConversationState, StateUpdate
from services.ConfigService import ConfigService


class EscalationChannel(str, Enum):
    """Supported escalation channels"""
    TICKET_SYSTEM = "ticket_system"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    NONE = "none"  # No escalation needed


class EscalationPriority(str, Enum):
    """Escalation priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EscalationDecision(BaseModel):
    """Escalation decision model"""
    should_escalate: bool = Field(..., description="Whether escalation is needed")
    channel: EscalationChannel = Field(..., description="Escalation channel to use")
    priority: EscalationPriority = Field(..., description="Escalation priority")
    reason: str = Field(..., description="Reason for escalation")
    confidence: float = Field(default=1.0, description="Decision confidence")


class EscalationAgent(BaseAgent):
    """
    Determines when and how to escalate conversations.

    This agent is responsible for:
    1. Evaluating if escalation is needed based on sentiment and policy
    2. Determining the appropriate escalation channel
    3. Calculating escalation priority
    4. Executing escalation via tools if configured
    5. Returning escalation state for ResponseAgent

    State Dependencies:
        - sentiment: Detected sentiment from SentimentAgent
        - sentiment_confidence: Sentiment confidence/urgency score
        - sentiment_raw: Raw sentiment data including toxicity_flag
        - policy_results: Policy evaluation results
        - policy_action: Recommended business action from PolicyAgent
        - intent: Detected intent for context
        - tenant_id: For tenant-specific escalation rules

    State Updates:
        - escalation_triggered: Whether escalation was triggered
        - escalation_channel: Channel used for escalation
        - escalation_details: Full escalation details (reason, priority, recipient, etc.)
        - escalation_raw: Raw escalation data for debugging

    Escalation Triggers:
        1. Sentiment-based:
           - angry sentiment with threshold >= 0.8
           - frustrated sentiment with threshold >= 0.6
           - toxicity_flag = True
           - sentiment_confidence (urgency) > 0.9

        2. Policy-based:
           - Policy indicates escalate_if_sentiment_above threshold exceeded
           - Policy has blocked_sentiments matching current sentiment
           - Policy requires escalation for specific intent

    Escalation Channels:
        - ticket_system: Create support ticket (default)
        - email: Send escalation email
        - slack: Post to Slack channel
        - webhook: Call external webhook
    """

    def __init__(
        self,
        llm_client,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tool_registry = None,
        config_service: Optional[ConfigService] = None,
        auto_escalate: bool = False
    ):
        """
        Initialize EscalationAgent.

        Args:
            llm_client: LLM client for inference
            system_prompt: Custom system prompt (optional)
            config: Additional configuration
            tool_registry: Tool registry for escalation tools
            config_service: ConfigService for loading escalation rules
            auto_escalate: Whether to automatically execute escalation via tools
        """
        super().__init__(
            llm_client=llm_client,
            system_prompt=system_prompt,
            config=config,
            tool_registry=tool_registry
        )
        self.config_service = config_service
        self.auto_escalate = auto_escalate

        # Escalation configuration (can be overridden by tenant config)
        self.escalation_config = config.get("escalation", {}) if config else {}

        # Priority thresholds (can be tenant-specific)
        self.priority_thresholds = self.escalation_config.get(
            "priority_thresholds",
            {
                "angry": EscalationPriority.CRITICAL,
                "frustrated": EscalationPriority.HIGH,
                "neutral": EscalationPriority.LOW,
                "positive": EscalationPriority.LOW
            }
        )

        # Channel mapping by priority
        self.channel_mapping = self.escalation_config.get(
            "channel_mapping",
            {
                EscalationPriority.CRITICAL: EscalationChannel.SLACK,  # Immediate notification
                EscalationPriority.HIGH: EscalationChannel.TICKET_SYSTEM,
                EscalationPriority.MEDIUM: EscalationChannel.TICKET_SYSTEM,
                EscalationPriority.LOW: EscalationChannel.NONE
            }
        )

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are an Escalation Agent. Your role is to determine when and how to escalate conversations to human agents.

## Your Responsibilities

1. **Evaluate**: Assess if escalation is needed based on sentiment, toxicity, and policy rules
2. **Prioritize**: Determine the appropriate priority level (LOW, MEDIUM, HIGH, CRITICAL)
3. **Channel**: Select the best escalation channel for the situation
4. **Document**: Provide clear reasoning for the escalation decision

## Escalation Triggers

**Immediate Escalation (CRITICAL):**
- Toxic language detected (toxicity_flag = True)
- Urgency score above 0.9
- Angry sentiment with high confidence (> 0.8)

**High Priority Escalation:**
- Frustrated sentiment (> 0.6 confidence)
- Policy indicates blocked action for current sentiment
- Policy requires escalation for intent

**Standard Escalation:**
- Customer explicitly requests escalation
- Multiple failed resolution attempts

## Escalation Channels

- **ticket_system**: Create support ticket (most common)
- **email**: Send escalation email to appropriate team
- **slack**: Immediate notification for critical issues
- **webhook**: Custom external integration

## Output Format

Respond with "ESCALATE: [channel]: [priority]: [reason]" or "NO_ESCALATION" if not needed.

Example:
- ESCALATE: ticket_system: high: Customer is frustrated and policy blocks refund
- ESCALATE: slack: critical: Toxic language detected, immediate human intervention required
- NO_ESCALATION"""

    async def process(self, input_data: Dict[str, Any], **kwargs) -> StateUpdate:
        """
        Process conversation state and determine if escalation is needed.

        Args:
            input_data: Dict with state field containing ConversationState
            **kwargs: Additional parameters

        Returns:
            StateUpdate with escalation fields populated
        """
        print(f"[DEBUG] EscalationAgent.process called")

        # Handle both ConversationState and dict input
        if isinstance(input_data, ConversationState):
            state = input_data
        elif isinstance(input_data, dict) and "state" in input_data:
            state = input_data["state"]
        else:
            # Legacy dict format
            state = self._dict_to_state(input_data)

        # Make escalation decision
        decision = await self._make_escalation_decision(state)

        print(f"[DEBUG] Escalation decision: {decision}")

        # Execute escalation via tools if auto_escalate is enabled
        escalation_result = None
        if decision.should_escalate and self.auto_escalate and decision.channel != EscalationChannel.NONE:
            escalation_result = await self._execute_escalation(decision, state)

        # Build escalation details
        escalation_details = {
            "reason": decision.reason,
            "priority": decision.priority.value,
            "channel": decision.channel.value,
            "timestamp": datetime.utcnow().isoformat(),
            "triggers": self._get_active_triggers(state),
            "sentiment": state.sentiment,
            "sentiment_confidence": state.sentiment_confidence,
            "intent": state.intent,
            "policy_action": state.policy_action
        }

        # Add escalation result if executed
        if escalation_result:
            escalation_details["execution_result"] = escalation_result

        # Create state update
        return StateUpdate(
            escalation_triggered=decision.should_escalate,
            escalation_channel=decision.channel.value if decision.should_escalate else None,
            escalation_details=escalation_details,
            escalation_raw={
                "decision": decision.dict(),
                "active_triggers": self._get_active_triggers(state),
                "sentiment_config": self._get_sentiment_escalation_config(state),
                "policy_config": self._get_policy_escalation_config(state)
            }
        )

    async def _make_escalation_decision(self, state: ConversationState) -> EscalationDecision:
        """
        Make escalation decision based on state.

        Args:
            state: Current conversation state

        Returns:
            EscalationDecision with recommendation
        """
        triggers = self._get_active_triggers(state)

        if not triggers:
            return EscalationDecision(
                should_escalate=False,
                channel=EscalationChannel.NONE,
                priority=EscalationPriority.LOW,
                reason="No escalation triggers detected",
                confidence=1.0
            )

        # Determine priority based on triggers
        priority = self._calculate_priority(state, triggers)

        # Determine channel based on priority
        channel = self.channel_mapping.get(priority, EscalationChannel.TICKET_SYSTEM)

        # Build reason
        reason = self._build_escalation_reason(state, triggers, priority)

        return EscalationDecision(
            should_escalate=True,
            channel=channel,
            priority=priority,
            reason=reason,
            confidence=self._calculate_confidence(state, triggers)
        )

    def _get_active_triggers(self, state: ConversationState) -> List[str]:
        """
        Get list of active escalation triggers.

        Args:
            state: Current conversation state

        Returns:
            List of active trigger names
        """
        triggers = []

        # Check sentiment raw data
        sentiment_raw = state.sentiment_raw or {}

        # Toxicity trigger
        if sentiment_raw.get("toxicity_flag"):
            triggers.append("toxicity_detected")

        # High urgency trigger
        if state.sentiment_confidence > 0.9:
            triggers.append("high_urgency")

        # Sentiment-based triggers
        if state.sentiment == "angry" and state.sentiment_confidence >= 0.8:
            triggers.append("angry_sentiment_high_confidence")
        elif state.sentiment == "angry":
            triggers.append("angry_sentiment")

        if state.sentiment == "frustrated" and state.sentiment_confidence >= 0.6:
            triggers.append("frustrated_sentiment_threshold_met")

        # Policy-based triggers
        if state.policy_results:
            # Check if sentiment exceeds policy threshold
            escalate_threshold = state.policy_results.get("escalate_if_sentiment_above")
            if escalate_threshold and state.sentiment_confidence >= escalate_threshold:
                triggers.append("policy_escalation_threshold_exceeded")

            # Check if sentiment is blocked by policy
            blocked_sentiments = state.policy_results.get("blocked_sentiments", [])
            if state.sentiment in blocked_sentiments:
                triggers.append("policy_blocked_sentiment")

            # Check if policy requires escalation
            if state.policy_results.get("requires_escalation"):
                triggers.append("policy_requires_escalation")

        return triggers

    def _calculate_priority(
        self,
        state: ConversationState,
        triggers: List[str]
    ) -> EscalationPriority:
        """
        Calculate escalation priority based on state and triggers.

        Args:
            state: Current conversation state
            triggers: Active escalation triggers

        Returns:
            EscalationPriority level
        """
        # Critical triggers
        if "toxicity_detected" in triggers:
            return EscalationPriority.CRITICAL
        if "high_urgency" in triggers:
            return EscalationPriority.CRITICAL
        if "angry_sentiment_high_confidence" in triggers:
            return EscalationPriority.CRITICAL

        # High priority triggers
        if "angry_sentiment" in triggers:
            return EscalationPriority.HIGH
        if "frustrated_sentiment_threshold_met" in triggers:
            return EscalationPriority.HIGH
        if "policy_escalation_threshold_exceeded" in triggers:
            return EscalationPriority.HIGH

        # Medium priority triggers
        if "policy_blocked_sentiment" in triggers:
            return EscalationPriority.MEDIUM

        # Default to low priority
        return EscalationPriority.LOW

    def _build_escalation_reason(
        self,
        state: ConversationState,
        triggers: List[str],
        priority: EscalationPriority
    ) -> str:
        """
        Build human-readable escalation reason.

        Args:
            state: Current conversation state
            triggers: Active escalation triggers
            priority: Calculated priority

        Returns:
            Human-readable reason string
        """
        parts = []

        # Add sentiment info
        if state.sentiment:
            parts.append(f"Customer sentiment: {state.sentiment}")

        # Add confidence/urgency
        if state.sentiment_confidence > 0.7:
            parts.append(f"Urgency level: {state.sentiment_confidence:.2f}")

        # Add specific triggers
        if "toxicity_detected" in triggers:
            parts.append("Toxic language detected")

        # Add policy info
        if state.policy_action:
            parts.append(f"Policy action: {state.policy_action}")

        # Add priority
        parts.append(f"Priority: {priority.value.upper()}")

        return ", ".join(parts)

    def _calculate_confidence(
        self,
        state: ConversationState,
        triggers: List[str]
    ) -> float:
        """
        Calculate confidence in escalation decision.

        Args:
            state: Current conversation state
            triggers: Active escalation triggers

        Returns:
            Confidence score (0-1)
        """
        if not triggers:
            return 0.0

        # High confidence for critical triggers
        if "toxicity_detected" in triggers or "high_urgency" in triggers:
            return 1.0

        # Base confidence on sentiment confidence
        base_confidence = state.sentiment_confidence

        # Boost confidence for multiple triggers
        trigger_boost = min(len(triggers) * 0.1, 0.2)

        return min(base_confidence + trigger_boost, 1.0)

    async def _execute_escalation(
        self,
        decision: EscalationDecision,
        state: ConversationState
    ) -> Optional[Dict[str, Any]]:
        """
        Execute escalation via appropriate tool.

        Args:
            decision: Escalation decision
            state: Current conversation state

        Returns:
            Execution result dict or None
        """
        if not self.tool_registry:
            print("[DEBUG] No tool_registry configured, skipping escalation execution")
            return None

        # Tool name mapping
        tool_map = {
            EscalationChannel.TICKET_SYSTEM: "TicketCreationTool",
            EscalationChannel.EMAIL: "EmailTool",
            EscalationChannel.SLACK: "SlackTool",
            EscalationChannel.WEBHOOK: "WebhookTool"
        }

        tool_name = tool_map.get(decision.channel)
        if not tool_name:
            print(f"[DEBUG] No tool configured for channel: {decision.channel}")
            return None

        if not self.can_use_tool(tool_name):
            print(f"[DEBUG] Agent doesn't have permission to use {tool_name}")
            return None

        # Build escalation payload
        payload = {
            "tenant_id": state.tenant_id,
            "session_id": state.session_id,
            "priority": decision.priority.value,
            "reason": decision.reason,
            "customer_message": state.user_message,
            "sentiment": state.sentiment,
            "intent": state.intent,
            "escalation_details": decision.dict()
        }

        try:
            print(f"[DEBUG] Executing escalation via {tool_name}")
            result = await self.use_tool(tool_name, payload)
            return {
                "tool": tool_name,
                "status": "success",
                "result": result.data if hasattr(result, "data") else result
            }
        except Exception as e:
            print(f"[DEBUG] Escalation execution failed: {str(e)}")
            return {
                "tool": tool_name,
                "status": "failed",
                "error": str(e)
            }

    def _get_sentiment_escalation_config(self, state: ConversationState) -> Dict[str, Any]:
        """Get sentiment-based escalation configuration."""
        # This would load from ConfigService in production
        return {
            "sentiment": state.sentiment,
            "confidence": state.sentiment_confidence,
            "threshold_for_escalation": 0.8 if state.sentiment == "angry" else 0.6
        }

    def _get_policy_escalation_config(self, state: ConversationState) -> Dict[str, Any]:
        """Get policy-based escalation configuration."""
        if not state.policy_results:
            return {}

        return {
            "escalate_if_sentiment_above": state.policy_results.get("escalate_if_sentiment_above"),
            "blocked_sentiments": state.policy_results.get("blocked_sentiments", []),
            "requires_escalation": state.policy_results.get("requires_escalation", False)
        }

    def _dict_to_state(self, data: Dict[str, Any]) -> Any:
        """
        Convert dict to state-like object for backward compatibility.

        Args:
            data: Dictionary with state fields

        Returns:
            Simple object with state attributes
        """
        class SimpleState:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        return SimpleState(
            tenant_id=data.get("tenant_id", "default"),
            session_id=data.get("session_id", ""),
            user_message=data.get("user_message", ""),
            sentiment=data.get("sentiment"),
            sentiment_confidence=data.get("sentiment_confidence", 0.0),
            sentiment_raw=data.get("sentiment_raw"),
            policy_results=data.get("policy_results", {}),
            policy_action=data.get("policy_action"),
            intent=data.get("intent"),
            add_error=lambda *args: None
        )

    @classmethod
    def get_agent_info(cls) -> Dict[str, Any]:
        """
        Get agent information.

        Returns:
            Dictionary with agent details
        """
        return {
            "name": "EscalationAgent",
            "description": "Determines when and how to escalate conversations to human agents",
            "input_fields": [
                "sentiment", "sentiment_confidence", "sentiment_raw",
                "policy_results", "policy_action", "intent"
            ],
            "output_fields": [
                "escalation_triggered", "escalation_channel", "escalation_details"
            ],
            "state_modifications": [
                "escalation_triggered", "escalation_channel", "escalation_details", "escalation_raw"
            ],
            "escalation_triggers": [
                "toxicity_detected",
                "high_urgency (> 0.9)",
                "angry_sentiment (> 0.8)",
                "frustrated_sentiment (> 0.6)",
                "policy_escalation_threshold_exceeded",
                "policy_blocked_sentiment"
            ],
            "supported_channels": [
                "ticket_system", "email", "slack", "webhook"
            ]
        }
