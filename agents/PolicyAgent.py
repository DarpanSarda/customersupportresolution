# agents/policy_agent.py

from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.sections import PolicyModel, PolicyViolation
from models.patch import Patch
from typing import Dict, List, Any


class PolicyAgent(BaseAgent):
    """
    Evaluates customer requests against configurable business policies.

    Reads:
    - understanding.intent: To match policy conditions
    - understanding.sentiment: To check sentiment-based restrictions
    - conversation.latest_message: To analyze request content
    - context: To get tenant_id, user profile

    Writes:
    - policy.compliant: Overall compliance status
    - policy.applicable_policies: List of policies evaluated
    - policy.violations: List of violations found
    - policy.restrictions: Restrictions to apply
    - policy.reason: Explanation

    All policies are config-driven - no hardcoded business rules.
    """

    agent_name = "PolicyAgent"
    allowed_section = "policy"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """
        Evaluate request against all configured policies.

        Returns Patch with policy evaluation results.
        """
        # -------------------------------------------------
        # 1️⃣ Check if policy evaluation is enabled
        # -------------------------------------------------
        if not self.config_loader.is_policy_enabled():
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "compliant": True,
                    "applicable_policies": [],
                    "violations": [],
                    "restrictions": [],
                    "reason": "Policy evaluation disabled"
                }
            )

        # -------------------------------------------------
        # 2️⃣ Extract data from state
        # -------------------------------------------------
        understanding = state.get("understanding", {})
        intent_data = understanding.get("intent", {})
        intent_name = intent_data.get("name")

        sentiment_data = understanding.get("sentiment", {})
        sentiment_label = sentiment_data.get("label", "NEUTRAL")

        context_data = state.get("context", {})
        tenant_id = context_data.get("tenant_id", context.tenant_id if context else "default")

        # -------------------------------------------------
        # 3️⃣ Get all policies from config
        # -------------------------------------------------
        all_policies = self.config_loader.get_policies()

        # -------------------------------------------------
        # 4️⃣ Evaluate each policy
        # -------------------------------------------------
        applicable_policies = []
        violations = []
        restrictions = []

        for policy_name, policy_config in all_policies.items():
            evaluation = self._evaluate_policy(
                policy_name=policy_name,
                policy_config=policy_config,
                intent_name=intent_name,
                sentiment_label=sentiment_label,
                tenant_id=tenant_id
            )

            if evaluation["applicable"]:
                applicable_policies.append(policy_name)

                if evaluation["violated"]:
                    violations.append(PolicyViolation(
                        policy_name=policy_name,
                        severity=evaluation["severity"],
                        reason=evaluation["reason"]
                    ))
                    restrictions.extend(evaluation.get("restrictions", []))

        # -------------------------------------------------
        # 5️⃣ Determine overall compliance
        # -------------------------------------------------
        is_compliant = len(violations) == 0

        if not is_compliant:
            # Check if any critical violations
            has_critical = any(v.severity == "CRITICAL" for v in violations)
            if has_critical:
                reason = f"Critical policy violations detected: {[v.policy_name for v in violations if v.severity == 'CRITICAL']}"
            else:
                reason = f"Policy violations detected: {[v.policy_name for v in violations]}"
        else:
            if applicable_policies:
                reason = f"All {len(applicable_policies)} applicable policies passed"
            else:
                reason = "No policies applicable to this request"

        # -------------------------------------------------
        # 6️⃣ Return Patch
        # -------------------------------------------------
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=1.0,  # Policy evaluation is deterministic
            changes={
                "compliant": is_compliant,
                "applicable_policies": applicable_policies,
                "violations": [v.model_dump() for v in violations],
                "restrictions": restrictions,
                "reason": reason
            }
        )

    def _evaluate_policy(
        self,
        policy_name: str,
        policy_config: dict,
        intent_name: str,
        sentiment_label: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single policy against the request.

        Returns:
            dict with keys: applicable, violated, severity, reason, restrictions
        """
        conditions = policy_config.get("conditions", {})
        actions = policy_config.get("actions_on_violation", {})

        # Check 1: Required intent match
        required_intents = conditions.get("required_intent", [])
        if required_intents:
            if intent_name not in required_intents:
                # Policy doesn't apply to this intent
                return {
                    "applicable": False,
                    "violated": False,
                    "severity": "NONE",
                    "reason": f"Policy {policy_name} not applicable for intent {intent_name}",
                    "restrictions": []
                }

        # Policy applies if we get here
        result = {
            "applicable": True,
            "violated": False,
            "severity": "NONE",
            "reason": f"Policy {policy_name} passed",
            "restrictions": []
        }

        # Check 2: Blocked sentiments
        blocked_sentiments = conditions.get("blocked_sentiments", [])
        if sentiment_label in blocked_sentiments:
            result.update({
                "violated": True,
                "severity": "HIGH",
                "reason": f"Sentiment {sentiment_label} triggers policy review for {policy_name}",
                "restrictions": [actions.get("restrict_response", "")]
            })
            return result

        # Check 3: Blocked tenant IDs
        blocked_tenants = conditions.get("blocked_for_tenant_ids", [])
        if tenant_id in blocked_tenants:
            result.update({
                "violated": True,
                "severity": "CRITICAL",
                "reason": f"Tenant {tenant_id} is blocked for {policy_name}",
                "restrictions": ["Service not available for this tenant"]
            })
            return result

        # Check 4: Require authentication
        if conditions.get("require_authentication"):
            if not self._is_authenticated(tenant_id):
                result.update({
                    "violated": True,
                    "severity": "HIGH",
                    "reason": f"Authentication required for {policy_name}",
                    "restrictions": ["Authentication required"]
                })
                return result

        # All checks passed
        return result

    def _is_authenticated(self, tenant_id: str) -> bool:
        """
        Check if tenant/session is authenticated.

        Placeholder for real authentication check.
        In production, would validate session token, API key, or user session.
        """
        # TODO: Implement real authentication check
        # For now, return True to allow policy evaluation to proceed
        return True
