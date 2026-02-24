# core/base_agent.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models.patch import Patch
from contextlib import contextmanager
import time


class AgentExecutionContext:
    """
    Immutable execution context passed by Orchestrator.
    """

    def __init__(
        self,
        trace_id: str,
        request_id: str,
        tenant_id: str,
        config_version: str,
        prompt_version: str,
        otel_span=None,
        langfuse_handler=None,
        logger=None,
    ):
        self.trace_id = trace_id
        self.request_id = request_id
        self.tenant_id = tenant_id
        self.config_version = config_version
        self.prompt_version = prompt_version
        self.otel_span = otel_span
        self.langfuse_handler = langfuse_handler
        self.logger = logger


class BaseAgent(ABC):
    """
    Enterprise-grade BaseAgent.

    Guarantees:
    - Stateless execution
    - Patch-only return
    - Section ownership enforcement
    - Confidence validation
    - Observability hook points
    - Prompt + Config version tracking
    """

    agent_name: str
    allowed_section: str

    def __init__(self, config: Dict[str, Any], prompt: str):
        self._config = config
        self._prompt = prompt

    # -----------------------------------------------------
    # PUBLIC EXECUTION ENTRYPOINT
    # -----------------------------------------------------

    def execute(
        self,
        state: Dict[str, Any],
        context: AgentExecutionContext,
    ) -> Patch:
        """
        Wrapper enforcing all enterprise constraints.
        """

        self._validate_statelessness(state)

        start_time = time.time()

        with self._start_observability_span(context):

            patch = self._run(state, context)

            if not isinstance(patch, Patch):
                raise TypeError(
                    f"{self.agent_name} must return Patch object"
                )

            self._validate_patch_integrity(patch)

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Inject mandatory metadata
            from models.patch import PatchMetadata

            metadata = PatchMetadata(
                execution_time_ms=execution_time_ms,
                config_version=context.config_version,
                prompt_version=context.prompt_version,
                trace_id=context.trace_id,
                request_id=context.request_id,
            )
            patch.metadata = metadata

            # Structured logging
            self._log_execution(context, patch)

            return patch

    # -----------------------------------------------------
    # ABSTRACT BUSINESS LOGIC
    # -----------------------------------------------------

    @abstractmethod
    def _run(
        self,
        state: Dict[str, Any],
        context: AgentExecutionContext,
    ) -> Patch:
        """
        Agent implementation.
        Must:
        - Not mutate state
        - Return valid Patch
        - Respect allowed_section
        """
        pass

    # -----------------------------------------------------
    # VALIDATION LAYER
    # -----------------------------------------------------

    def _validate_patch_integrity(self, patch: Patch):

        if patch.agent_name != self.agent_name:
            raise ValueError(
                f"Agent name mismatch. Expected {self.agent_name}"
            )

        if patch.target_section != self.allowed_section:
            raise ValueError(
                f"{self.agent_name} cannot write to {patch.target_section}"
            )

        if not (0.0 <= patch.confidence <= 1.0):
            raise ValueError(
                "Patch confidence must be between 0 and 1"
            )

    def _validate_statelessness(self, state: Dict[str, Any]):
        """
        Prevent accidental mutation patterns.
        Can be extended with hashing if needed.
        """
        if not isinstance(state, dict):
            raise TypeError("State must be dictionary")

    # -----------------------------------------------------
    # OBSERVABILITY HOOKS
    # -----------------------------------------------------

    @contextmanager
    def _start_observability_span(self, context: AgentExecutionContext):
        """
        Wrap execution with OpenTelemetry + Langfuse.
        """

        span = None

        if context.otel_span:
            span = context.otel_span.start_as_current_span(
                f"agent:{self.agent_name}"
            )

        try:
            if span:
                span.__enter__()

            # Langfuse trace hook
            if context.langfuse_handler:
                context.langfuse_handler.start_trace(
                    name=self.agent_name,
                    metadata={
                        "tenant_id": context.tenant_id,
                        "trace_id": context.trace_id,
                    }
                )

            yield

        finally:
            if context.langfuse_handler:
                context.langfuse_handler.end_trace()

            if span:
                span.__exit__(None, None, None)

    # -----------------------------------------------------
    # LOGGING
    # -----------------------------------------------------

    def _log_execution(
        self,
        context: AgentExecutionContext,
        patch: Patch
    ):
        if context.logger:
            context.logger.info({
                "event": "agent_execution",
                "agent": self.agent_name,
                "trace_id": context.trace_id,
                "request_id": context.request_id,
                "confidence": patch.confidence,
                "target_section": patch.target_section,
            })