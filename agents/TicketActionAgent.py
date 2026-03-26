"""
TicketActionAgent - Executes business actions determined by PolicyAgent.

This agent is responsible for taking action based on policy decisions:
- Creating support tickets
- Processing refund requests
- Sending email notifications
- Executing API calls to external systems

Follows the Agent Contract pattern with StateUpdate for results.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4
from core.BaseAgent import BaseAgent
from core.ConversationState import StateUpdate
from pydantic import BaseModel, Field


class ActionExecutionResult(BaseModel):
    """Result of executing a business action."""
    action_executed: bool = Field(False, description="Whether the action was executed")
    action_type: Optional[str] = Field(None, description="Type of action executed")
    action_id: Optional[str] = Field(None, description="Unique ID for the executed action")
    status: str = Field("pending", description="Status: pending, completed, failed, blocked")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action execution details")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class TicketActionAgent(BaseAgent):
    """
    Ticket/Action execution agent.

    Executes business actions based on policy decisions:
    - Creates tickets in the database
    - Processes refunds
    - Sends notifications
    - Calls external APIs

    Returns a StateUpdate with action execution results.
    """

    def __init__(
        self,
        llm_client=None,
        system_prompt: Optional[str] = None,
        config_service=None,
        db_manager=None,
        auto_execute: bool = False
    ):
        """
        Initialize TicketActionAgent.

        Args:
            llm_client: LLM client for generating action content
            system_prompt: System prompt for the agent
            config_service: Configuration service for tenant settings
            db_manager: Database manager for ticket creation
            auto_execute: If False, only validates and prepares. If True, executes actions.
        """
        super().__init__(llm_client=llm_client, system_prompt=system_prompt)
        self.config_service = config_service
        self.db_manager = db_manager
        self.auto_execute = auto_execute

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return "You are a Ticket Action Agent. Execute business actions based on policy decisions."

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "TicketActionAgent"

    def get_input_fields(self) -> List[str]:
        """Get required input fields."""
        return [
            "policy_action",
            "policy_results",
            "intent",
            "sentiment",
            "escalation_triggered",
            "user_message",
            "tenant_id",
            "session_id"
        ]

    def get_output_fields(self) -> List[str]:
        """Get output fields."""
        return [
            "action_executed",
            "action_type",
            "action_id",
            "action_status",
            "action_details",
            "action_raw"
        ]

    async def process(
        self,
        input_data: Dict[str, Any],
        **kwargs
    ) -> StateUpdate:
        """
        Process ticket action request.

        Args:
            input_data: Dictionary containing conversation state
            **kwargs: Additional parameters

        Returns:
            StateUpdate with action execution results
        """
        # Extract key fields from input (support both state object and dict)
        state = input_data.get("state")
        if state:
            # State object passed
            policy_action = state.policy_action
            policy_results = state.policy_results or {}
            intent = state.intent
            sentiment = state.sentiment
            escalation_triggered = state.escalation_triggered
            user_message = state.user_message
            tenant_id = state.tenant_id
            session_id = state.session_id
        else:
            # Legacy dict format
            policy_action = input_data.get("policy_action")
            policy_results = input_data.get("policy_results", {})
            intent = input_data.get("intent")
            sentiment = input_data.get("sentiment")
            escalation_triggered = input_data.get("escalation_triggered", False)
            user_message = input_data.get("user_message")
            tenant_id = input_data.get("tenant_id", "default")
            session_id = input_data.get("session_id")

        # Check if execution is allowed
        allow_execution = policy_results.get("allow_execution", False)
        blocked_sentiments = policy_results.get("blocked_sentiments", [])
        is_blocked = sentiment in blocked_sentiments

        print(f"[DEBUG] TicketActionAgent: action={policy_action}, allow={allow_execution}, blocked={is_blocked}")

        # Initialize result
        result = ActionExecutionResult()

        # Determine if action should be executed
        should_execute = (
            policy_action and
            allow_execution and
            not is_blocked and
            self.auto_execute
        )

        if not policy_action:
            result.status = "no_action"
            result.details["reason"] = "No policy action specified"

        elif is_blocked:
            result.status = "blocked"
            result.action_type = policy_action
            result.details["reason"] = f"Action blocked due to sentiment: {sentiment}"

        elif not allow_execution:
            result.status = "not_allowed"
            result.action_type = policy_action
            result.details["reason"] = "Policy does not allow automatic execution"

        elif should_execute:
            # Execute the action
            execution_result = await self._execute_action(
                action_type=policy_action,
                intent=intent,
                sentiment=sentiment,
                user_message=user_message,
                tenant_id=tenant_id,
                session_id=session_id,
                policy_results=policy_results
            )

            result.action_executed = True
            result.action_type = policy_action
            result.action_id = execution_result.get("action_id", str(uuid4()))
            result.status = execution_result.get("status", "completed")
            result.details = execution_result

        else:
            # Action validated but not executed (auto_execute=False)
            result.status = "validated"
            result.action_type = policy_action
            result.action_id = str(uuid4())
            result.details["reason"] = "Action validated but not executed (auto_execute=False)"
            result.details["action_would_be"] = policy_action

        print(f"[DEBUG] TicketActionAgent result: executed={result.action_executed}, status={result.status}")

        # Return StateUpdate
        return StateUpdate(
            action_executed=result.action_executed,
            action_type=result.action_type,
            action_id=result.action_id,
            action_status=result.status,
            action_details=result.details,
            action_raw=result.model_dump()
        )

    async def _execute_action(
        self,
        action_type: str,
        intent: str,
        sentiment: str,
        user_message: str,
        tenant_id: str,
        session_id: str,
        policy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a business action.

        Args:
            action_type: Type of action to execute (CREATE_TICKET, PROCESS_REFUND, etc.)
            intent: User's intent
            sentiment: User's sentiment
            user_message: Original user message
            tenant_id: Tenant ID
            session_id: Session ID
            policy_results: Policy evaluation results

        Returns:
            Dictionary with execution result
        """
        print(f"[DEBUG] Executing action: {action_type}")

        # Generate action content using LLM
        action_content = await self._generate_action_content(
            action_type=action_type,
            intent=intent,
            sentiment=sentiment,
            user_message=user_message,
            policy_results=policy_results
        )

        # Execute based on action type
        if action_type == "CREATE_TICKET":
            return await self._create_ticket(
                tenant_id=tenant_id,
                session_id=session_id,
                intent=intent,
                sentiment=sentiment,
                user_message=user_message,
                action_content=action_content
            )

        elif action_type == "PROCESS_REFUND":
            return await self._process_refund(
                tenant_id=tenant_id,
                session_id=session_id,
                user_message=user_message,
                action_content=action_content,
                policy_results=policy_results
            )

        elif action_type == "SEND_EMAIL":
            return await self._send_email(
                tenant_id=tenant_id,
                session_id=session_id,
                intent=intent,
                action_content=action_content
            )

        elif action_type == "ESCALATE":
            # Escalation is handled by EscalationAgent
            return {
                "status": "delegated",
                "reason": "Escalation handled by EscalationAgent",
                "action_id": str(uuid4())
            }

        else:
            return {
                "status": "unknown_action",
                "reason": f"Unknown action type: {action_type}",
                "action_id": str(uuid4())
            }

    async def _generate_action_content(
        self,
        action_type: str,
        intent: str,
        sentiment: str,
        user_message: str,
        policy_results: Dict[str, Any]
    ) -> str:
        """
        Generate content for the action using LLM.

        Args:
            action_type: Type of action
            intent: User's intent
            sentiment: User's sentiment
            user_message: Original user message
            policy_results: Policy results

        Returns:
            Generated content
        """
        if not self.llm_client:
            # Return basic content if no LLM
            return f"Action: {action_type}, Intent: {intent}, Message: {user_message[:100]}"

        prompt = self._build_action_prompt(action_type, intent, sentiment, user_message, policy_results)

        try:
            response = await self.llm_client.generate_with_messages(
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            return response.content
        except Exception as e:
            print(f"[DEBUG] LLM content generation failed: {str(e)}")
            return f"Action: {action_type}, Intent: {intent}"

    def _build_action_prompt(
        self,
        action_type: str,
        intent: str,
        sentiment: str,
        user_message: str,
        policy_results: Dict[str, Any]
    ) -> str:
        """Build prompt for action content generation."""
        return f"""Generate a concise action description for the following:

Action Type: {action_type}
User Intent: {intent}
User Sentiment: {sentiment}
User Message: {user_message}

Generate a 1-2 sentence description of what action should be taken. Be specific and actionable.
"""

    async def _create_ticket(
        self,
        tenant_id: str,
        session_id: str,
        intent: str,
        sentiment: str,
        user_message: str,
        action_content: str
    ) -> Dict[str, Any]:
        """
        Create a support ticket.

        Args:
            tenant_id: Tenant ID
            session_id: Session ID
            intent: User's intent
            sentiment: User's sentiment
            user_message: User's message
            action_content: Generated action content

        Returns:
            Ticket creation result
        """
        ticket_id = str(uuid4())

        ticket_data = {
            "ticket_id": ticket_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "intent": intent,
            "sentiment": sentiment,
            "description": action_content or user_message,
            "user_message": user_message,
            "status": "open",
            "priority": self._calculate_priority(sentiment),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Store in database if available
        if self.db_manager:
            try:
                await self.db_manager.insert(
                    table="tickets",
                    data=ticket_data
                )
                print(f"[DEBUG] Ticket created in database: {ticket_id}")
            except Exception as e:
                print(f"[DEBUG] Failed to create ticket in DB: {str(e)}")
                # Continue anyway, ticket is still created in memory
        else:
            print(f"[DEBUG] No DB manager, ticket not persisted: {ticket_id}")

        return {
            "status": "completed",
            "action_id": ticket_id,
            "ticket_data": ticket_data,
            "message": f"Ticket {ticket_id} created successfully"
        }

    async def _process_refund(
        self,
        tenant_id: str,
        session_id: str,
        user_message: str,
        action_content: str,
        policy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a refund request.

        Args:
            tenant_id: Tenant ID
            session_id: Session ID
            user_message: User's message
            action_content: Generated action content
            policy_results: Policy results with required fields

        Returns:
            Refund processing result
        """
        refund_id = str(uuid4())

        # Check for required fields
        required_fields = policy_results.get("required_fields", [])
        missing_fields = policy_results.get("missing_fields", [])

        refund_data = {
            "refund_id": refund_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "description": action_content or user_message,
            "status": "pending_review" if missing_fields else "processing",
            "required_fields": required_fields,
            "missing_fields": missing_fields,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Store refund request
        if self.db_manager:
            try:
                await self.db_manager.insert(
                    table="refund_requests",
                    data=refund_data
                )
                print(f"[DEBUG] Refund request created: {refund_id}")
            except Exception as e:
                print(f"[DEBUG] Failed to create refund request: {str(e)}")

        message = f"Refund request {refund_id} created"
        if missing_fields:
            message += f" (missing fields: {', '.join(missing_fields)})"

        return {
            "status": "completed",
            "action_id": refund_id,
            "refund_data": refund_data,
            "message": message
        }

    async def _send_email(
        self,
        tenant_id: str,
        session_id: str,
        intent: str,
        action_content: str
    ) -> Dict[str, Any]:
        """
        Send an email notification.

        Args:
            tenant_id: Tenant ID
            session_id: Session ID
            intent: User's intent
            action_content: Email content

        Returns:
            Email sending result
        """
        email_id = str(uuid4())

        email_data = {
            "email_id": email_id,
            "tenant_id": tenant_id,
            "session_id": session_id,
            "intent": intent,
            "content": action_content,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Store email request
        if self.db_manager:
            try:
                await self.db_manager.insert(
                    table="email_queue",
                    data=email_data
                )
                print(f"[DEBUG] Email queued: {email_id}")
            except Exception as e:
                print(f"[DEBUG] Failed to queue email: {str(e)}")

        return {
            "status": "completed",
            "action_id": email_id,
            "email_data": email_data,
            "message": f"Email {email_id} queued for sending"
        }

    def _calculate_priority(self, sentiment: str) -> str:
        """Calculate ticket priority based on sentiment."""
        priority_map = {
            "angry": "critical",
            "frustrated": "high",
            "neutral": "normal",
            "positive": "low"
        }
        return priority_map.get(sentiment, "normal")

    @staticmethod
    def get_agent_info() -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "TicketActionAgent",
            "description": "Executes business actions based on policy decisions",
            "input_fields": [
                "policy_action",
                "policy_results",
                "intent",
                "sentiment",
                "escalation_triggered",
                "user_message",
                "tenant_id",
                "session_id"
            ],
            "output_fields": [
                "action_executed",
                "action_type",
                "action_id",
                "action_status",
                "action_details",
                "action_raw"
            ],
            "supported_actions": [
                "CREATE_TICKET",
                "PROCESS_REFUND",
                "SEND_EMAIL",
                "ESCALATE"
            ],
            "execution_modes": [
                "validate_only (auto_execute=False)",
                "auto_execute (auto_execute=True)"
            ]
        }
