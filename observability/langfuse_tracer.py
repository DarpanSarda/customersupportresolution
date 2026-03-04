"""
Langfuse integration for LLM observability.

Features:
- Automatic LLM call tracking
- Prompt/completion monitoring
- Token usage and cost tracking
- Session-based trace grouping
- Scoring and feedback support

Usage:
    tracer = LangfuseTracer(
        public_key="pk-xxx",
        secret_key="sk-yyy"
    )

    # Create observation
    observation = tracer.create_observation(
        trace_id="...",
        agent_name="IntentAgent",
        input_messages=[...],
        model="llama-3.3-71b-versatile"
    )

    # Complete observation
    tracer.complete_observation(
        observation_id=observation.id,
        output_content="...",
        usage={"input_tokens": 10, "output_tokens": 20}
    )
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LangfuseObservation:
    """Holds Langfuse observation data."""

    def __init__(
        self,
        observation_id: Optional[str],
        trace_id: str,
        agent_name: str,
        model: str
    ):
        self.id = observation_id
        self.trace_id = trace_id
        self.agent_name = agent_name
        self.model = model
        self.start_time = datetime.utcnow()


class LangfuseTracer:
    """
    Langfuse tracer for LLM observability.

    Langfuse provides:
    - Prompt/response tracking
    - Token usage and costs
    - Session grouping
    - User tracking
    - Scoring/feedback
    """

    # Model pricing (USD per 1M tokens) - approximate
    MODEL_PRICING = {
        "llama-3.3-71b-versatile": {"input": 0.0, "output": 0.0},  # Groq free tier
        "llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
        "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
        "gemma2-9b-it": {"input": 0.0, "output": 0.0},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    }

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
        sample_rate: float = 1.0,
        debug: bool = False
    ):
        """
        Initialize Langfuse tracer.

        Args:
            public_key: Langfuse public key (from env: LANGFUSE_PUBLIC_KEY)
            secret_key: Langfuse secret key (from env: LANGFUSE_SECRET_KEY)
            host: Langfuse host URL (default: cloud.langfuse.com)
            enabled: Enable/disable tracing
            sample_rate: Sampling rate (0.0 to 1.0)
            debug: Enable debug logging
        """
        import os

        self.public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
        self.host = host
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.debug = debug

        # Auto-disable if no credentials
        self.enabled = enabled and bool(self.public_key and self.secret_key)

        self._client: Optional[Any] = None

        if self.enabled:
            self._initialize()
        elif debug:
            logger.info("Langfuse tracing disabled (no credentials)")

    def _initialize(self):
        """Initialize Langfuse client."""
        try:
            from langfuse import Langfuse
            self._client = Langfuse(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host
            )
            if self.debug:
                logger.info(f"Langfuse initialized (host={self.host})")
        except ImportError:
            logger.warning(
                "langfuse package not installed. "
                "Install with: pip install langfuse"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"Langfuse initialization failed: {e}")
            self.enabled = False

    def create_observation(
        self,
        trace_id: str,
        agent_name: str,
        input_messages: List[Dict[str, str]],
        model: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_observation_id: Optional[str] = None
    ) -> Optional[LangfuseObservation]:
        """
        Create a new LLM observation (generation).

        Args:
            trace_id: Unique trace identifier
            agent_name: Name of the agent making the LLM call
            input_messages: Input messages [{role, content}]
            model: Model name/identifier
            metadata: Additional metadata
            session_id: Session identifier for grouping
            user_id: User identifier
            parent_observation_id: Parent observation ID (for nested calls)

        Returns:
            LangfuseObservation object or None if disabled/sampled out
        """
        if not self.enabled or not self._client:
            return None

        # Sample based on rate
        import random
        if random.random() > self.sample_rate:
            if self.debug:
                logger.debug(f"Langfuse observation sampled out (rate={self.sample_rate})")
            return None

        try:
            # Flatten messages for prompt
            prompt_content = self._format_messages(input_messages)

            # Create generation via Langfuse SDK
            generation = self._client.generation(
                trace_id=trace_id,
                name=f"{agent_name}_llm_call",
                model=model,
                input=prompt_content,
                metadata={
                    "agent": agent_name,
                    "message_count": len(input_messages),
                    **(metadata or {})
                },
                session_id=session_id,
                user_id=user_id,
                parent_observation_id=parent_observation_id
            )

            observation = LangfuseObservation(
                observation_id=getattr(generation, "id", None),
                trace_id=trace_id,
                agent_name=agent_name,
                model=model
            )

            if self.debug:
                logger.debug(
                    f"Langfuse observation created: {agent_name} "
                    f"(trace={trace_id}, model={model})"
                )

            return observation

        except Exception as e:
            logger.warning(f"Langfuse observation creation failed: {e}")
            return None

    def complete_observation(
        self,
        trace_id: str,
        observation_id: Optional[str],
        agent_name: str,
        model: str,
        output_content: str,
        usage: Optional[Dict[str, int]] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Complete an LLM observation with results.

        Args:
            trace_id: Trace identifier
            observation_id: Observation ID from create_observation
            agent_name: Agent name
            model: Model name
            output_content: LLM output content
            usage: Token usage {prompt_tokens, completion_tokens}
            latency_ms: Response time in milliseconds
            metadata: Additional metadata
        """
        if not self.enabled or not self._client:
            return

        if not observation_id:
            return

        try:
            # Calculate cost
            cost = None
            if usage:
                cost = self._calculate_cost(model, usage)

            # Update the generation
            # Note: Langfuse SDK v2 uses different API
            # We create a completion event
            self._client.generation(
                id=observation_id,
                trace_id=trace_id,
                name=f"{agent_name}_llm_call",
                model=model,
                output=output_content,
                usage=usage or {},
                latency_ms=latency_ms,
                metadata=metadata or {},
                completion_start_time=self._get_completion_start_time(latency_ms)
            )

            if self.debug:
                logger.debug(
                    f"Langfuse observation completed: {agent_name} "
                    f"(tokens={usage}, latency={latency_ms}ms, cost={cost})"
                )

        except Exception as e:
            logger.warning(f"Langfuse observation completion failed: {e}")

    def score_trace(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None
    ) -> None:
        """
        Add a score to a trace (for feedback).

        Args:
            trace_id: Trace identifier
            score: Score value (typically 0-1 or 1-5)
            comment: Optional comment
        """
        if not self.enabled or not self._client:
            return

        try:
            self._client.score(
                trace_id=trace_id,
                name="quality",
                value=score,
                comment=comment
            )
        except Exception as e:
            logger.warning(f"Langfuse score failed: {e}")

    def create_trace(
        self,
        trace_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        input_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional["LangfuseTrace"]:
        """
        Create a new trace explicitly.

        Note: In Langfuse v2, traces are created implicitly through observations.
        This method returns a LangfuseTrace wrapper that stores trace metadata
        for later use with observations.

        Args:
            trace_id: Unique trace identifier
            session_id: Session identifier
            user_id: User identifier
            tenant_id: Tenant identifier (added to metadata)
            input_message: Input message for the trace
            metadata: Trace metadata

        Returns:
            LangfuseTrace object for updating the trace
        """
        if not self.enabled:
            return None

        try:
            # Merge tenant_id into metadata
            trace_metadata = metadata or {}
            if tenant_id:
                trace_metadata["tenant_id"] = tenant_id

            # In Langfuse v2, traces are created implicitly through generations/scores
            # We store the metadata and let the first observation create the trace
            # If client exists, we can try to create a trace for context
            trace_obj = None
            if self._client:
                # Langfuse v2: trace() returns a Trace object that can be updated
                # The trace is created when you call update() on it
                if hasattr(self._client, "trace"):
                    trace_obj = self._client.trace(
                        id=trace_id,
                        session_id=session_id,
                        user_id=user_id,
                        input=input_message,
                        metadata=trace_metadata
                    )
                    # Create an explicit trace by updating it
                    if trace_obj and hasattr(trace_obj, "update"):
                        trace_obj.update()
                else:
                    # If trace() method not available, just store metadata
                    # The trace will be created implicitly with the first observation
                    trace_obj = None

            return LangfuseTrace(trace_obj)
        except Exception as e:
            logger.debug(f"Langfuse trace creation failed (non-critical): {e}")
            # Return a NoOp trace that won't fail on finalize
            return LangfuseTrace(None)

    def flush(self):
        """Flush any pending traces to Langfuse."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.warning(f"Langfuse flush failed: {e}")

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a single string."""
        return "\n".join([
            f"{msg.get('role', '').upper()}: {msg.get('content', '')}"
            for msg in messages
        ])

    def _calculate_cost(
        self,
        model: str,
        usage: Dict[str, int]
    ) -> Optional[float]:
        """Calculate cost based on token usage."""
        pricing = self.MODEL_PRICING.get(model)
        if not pricing:
            return None

        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def _get_completion_start_time(self, latency_ms: Optional[int]) -> Optional[datetime]:
        """Calculate completion start time from latency."""
        if latency_ms:
            return datetime.utcnow()
        return None


class LangfuseTrace:
    """
    Wrapper for Langfuse trace object to support finalization.

    Usage:
        trace = tracer.create_trace(trace_id, session_id, tenant_id, input_message)
        # ... do work ...
        trace.finalize(output="response", metadata={...})
        trace.finalize_error(error_message="error", error_type="TypeError")
    """

    def __init__(self, trace):
        self._trace = trace

    def finalize(
        self,
        output: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Finalize the trace with output and metadata."""
        if self._trace:
            try:
                update_data = {}
                if output:
                    update_data["output"] = output
                if metadata:
                    update_data["metadata"] = metadata
                self._trace.update(**update_data)
            except Exception as e:
                logger.warning(f"Langfuse trace finalization failed: {e}")

    def finalize_error(
        self,
        error_message: str,
        error_type: Optional[str] = None,
        level: str = "error"
    ):
        """Finalize the trace with error information."""
        if self._trace:
            try:
                self._trace.update(
                    output=error_message,
                    level=level,
                    metadata={"error_type": error_type} if error_type else {}
                )
            except Exception as e:
                logger.warning(f"Langfuse trace error finalization failed: {e}")


class LangfuseSpan:
    """
    Context manager for Langfuse observation tracking.

    Usage:
        with LangfuseSpan(tracer, trace_id, "IntentAgent", messages, model) as span:
            # LLM call here
            result = llm.generate(...)
            span.complete(result.content, result.usage)
    """

    def __init__(
        self,
        tracer: LangfuseTracer,
        trace_id: str,
        agent_name: str,
        input_messages: List[Dict[str, str]],
        model: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.tracer = tracer
        self.trace_id = trace_id
        self.agent_name = agent_name
        self.input_messages = input_messages
        self.model = model
        self.session_id = session_id
        self.user_id = user_id
        self.observation: Optional[LangfuseObservation] = None

    def __enter__(self) -> "LangfuseSpan":
        self.observation = self.tracer.create_observation(
            trace_id=self.trace_id,
            agent_name=self.agent_name,
            input_messages=self.input_messages,
            model=self.model,
            session_id=self.session_id,
            user_id=self.user_id
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and self.observation:
            # Log error in Langfuse
            self.tracer.complete_observation(
                trace_id=self.trace_id,
                observation_id=self.observation.id,
                agent_name=self.agent_name,
                model=self.model,
                output_content=f"ERROR: {exc_val}",
                usage=None,
                metadata={"error": str(exc_val)}
            )
        return False  # Don't suppress exceptions

    def complete(
        self,
        output_content: str,
        usage: Optional[Dict[str, int]] = None,
        latency_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Complete the observation with results."""
        if self.observation:
            self.tracer.complete_observation(
                trace_id=self.trace_id,
                observation_id=self.observation.id,
                agent_name=self.agent_name,
                model=self.model,
                output_content=output_content,
                usage=usage,
                latency_ms=latency_ms,
                metadata=metadata
            )
