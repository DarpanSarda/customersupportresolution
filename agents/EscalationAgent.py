"""Escalation Agent for execution of escalation workflow.

Executes escalation workflow AFTER DecisionAgent determines escalation is required.

Responsibilities:
- Build escalation payload with full context
- Determine escalation level (LOW, MEDIUM, HIGH, CRITICAL)
- Select escalation channel (tenant-configured)
- Trigger escalation tool via business_action
- Write escalation metadata to state

NOT responsible for:
- Deciding whether to escalate (DecisionAgent's job)
- Analyzing sentiment (SentimentAgent's job)
- Evaluating policy (PolicyAgent's job)
- Calling tools directly (ToolExecutionAgent's job)
"""

import time
from typing import Dict, Any, Optional
from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch
from models.sections import EscalationSchema
import logging

logger = logging.getLogger(__name__)


class EscalationAgent(BaseAgent):
    """
    Executes escalation workflow after DecisionAgent determines escalation.

    Reads:
    - decision.action: Must be "ESCALATE"
    - decision.route: Escalation trigger reason
    - decision.reason: Human-readable escalation reason
    - understanding.sentiment: For priority calculation
    - policy.priority: Base priority from policy evaluation
    - context.tenant_id: For tenant-specific escalation config
    - lifecycle.session_id: For tracking

    Writes:
    - escalation.reason: Why escalation was triggered
    - escalation.priority: Priority level (LOW, MEDIUM, HIGH, CRITICAL)
    - escalation.status: Escalation status
    - escalation.channel: Escalation channel
    - escalation.business_action: Action for ToolExecutionAgent
    - escalation.summary: Context summary for human agent
    - escalation.escalated_at: Timestamp
    """

    agent_name = "EscalationAgent"
    allowed_section = "escalation"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")

        # Escalation configuration
        self.escalation_config = config.get("escalation_config", {})
        self.priority_thresholds = self.escalation_config.get("priority_thresholds", {})
        self.default_channel = self.escalation_config.get("default_channel", "ticket_system")
        self.enabled = self.escalation_config.get("enabled", True)

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Execute escalation workflow.

        Step 1: Verify decision.action == "ESCALATE"
        Step 2: Build escalation context
        Step 3: Determine priority level
        Step 4: Select channel from tenant config
        Step 5: Write escalation metadata
        """
        # -------------------------------------------------
        # 1️⃣ Verify escalation decision
        # -------------------------------------------------
        decision = state.get("decision", {})

        if decision.get("action") != "ESCALATE":
            # Not an escalation - return empty patch
            return self._empty_patch("Not an escalation decision")

        # -------------------------------------------------
        # 2️⃣ Extract escalation context
        # -------------------------------------------------
        route = decision.get("route", "UNKNOWN")
        reason = decision.get("reason", "Escalation required")

        understanding = state.get("understanding", {})
        sentiment = understanding.get("sentiment", {})
        sentiment_label = sentiment.get("label", "NEUTRAL")

        policy = state.get("policy", {})
        policy_priority = policy.get("priority", "normal")

        context_data = state.get("context", {})
        tenant_id = context_data.get("tenant_id", "default")

        lifecycle = state.get("lifecycle", {})
        session_id = lifecycle.get("session_id")

        # -------------------------------------------------
        # 3️⃣ Determine escalation priority
        # -------------------------------------------------
        priority = self._determine_priority(
            route=route,
            sentiment_label=sentiment_label,
            policy_priority=policy_priority
        )

        # -------------------------------------------------
        # 4️⃣ Select escalation channel
        # -------------------------------------------------
        channel = self._get_channel(tenant_id)

        # -------------------------------------------------
        # 5️⃣ Build escalation summary for human agent
        # -------------------------------------------------
        summary = self._build_summary(
            route=route,
            reason=reason,
            sentiment_label=sentiment_label,
            user_input=understanding.get("input", {}).get("raw_text", "")
        )

        # -------------------------------------------------
        # 6️⃣ Build escalation patch
        # -------------------------------------------------
        escalated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "reason": route,
                "priority": priority,
                "status": "initiated",
                "channel": channel,
                "business_action": "CREATE_ESCALATION_TICKET",
                "tenant_id": tenant_id,
                "session_id": session_id,
                "summary": summary,
                "escalated_at": escalated_at
            }
        )

    def _determine_priority(
        self,
        route: str,
        sentiment_label: str,
        policy_priority: str
    ) -> str:
        """Determine escalation priority level.

        Priority mapping (config-driven):
        - Sentiment-based: ANGGY → CRITICAL, FRUSTRATED → HIGH
        - Policy-based: Uses policy.priority
        - Route-based: Certain routes map to specific priorities

        Returns: LOW, MEDIUM, HIGH, or CRITICAL
        """
        # Check sentiment-based priority first
        if sentiment_label in self.priority_thresholds:
            return self.priority_thresholds[sentiment_label]

        # Map policy priority to escalation priority
        policy_priority_map = {
            "low": "LOW",
            "normal": "MEDIUM",
            "high": "HIGH",
            "urgent": "CRITICAL"
        }
        if policy_priority in policy_priority_map:
            return policy_priority_map[policy_priority]

        # Route-based priority fallback
        route_priority_map = {
            "HIGH_NEGATIVE_SENTIMENT": "HIGH",
            "POLICY_ESCALATION": "HIGH",
            "LOW_CONFIDENCE": "MEDIUM",
            "NO_TOOL_MAPPING": "MEDIUM",
            "COMPLIANCE_REQUIRED": "CRITICAL",
            "VIP_CUSTOMER": "HIGH"
        }
        return route_priority_map.get(route, "MEDIUM")

    def _get_channel(self, tenant_id: str) -> str:
        """Get escalation channel for tenant.

        Tenant config defines:
        {
          "escalation": {
            "channel": "ticket_system" | "email" | "slack" | "webhook"
          }
        }
        """
        # Check tenant-specific escalation config
        if self.config_loader:
            tenant_escalation_config = self.config_loader.get_tenant_escalation_config(tenant_id)
            if tenant_escalation_config:
                return tenant_escalation_config.get("channel", self.default_channel)

        return self.default_channel

    def _build_summary(
        self,
        route: str,
        reason: str,
        sentiment_label: str,
        user_input: str
    ) -> str:
        """Build escalation summary for human agent.

        Includes:
        - Trigger reason
        - Sentiment context
        - User input (truncated)
        """
        # Truncate user input if too long
        input_preview = user_input[:200] + "..." if len(user_input) > 200 else user_input

        summary_parts = [
            f"Trigger: {route}",
            f"Sentiment: {sentiment_label}",
            f"Reason: {reason}",
            f"User Input: {input_preview}"
        ]

        return " | ".join(summary_parts)

    def _empty_patch(self, reason: str) -> Patch:
        """Return empty patch when not escalating."""
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,
            changes={
                "status": "skipped",
                "reason": reason
            }
        )
