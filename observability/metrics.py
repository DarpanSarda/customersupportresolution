"""
Prometheus metrics collector for monitoring.

Features:
- Counter for request counts
- Histogram for latency tracking
- Gauge for concurrent requests
- Summary for percentile tracking

Usage:
    metrics = MetricsCollector()

    metrics.record_agent_execution(
        agent_name="IntentAgent",
        tenant_id="amazon",
        status="success",
        latency_seconds=0.234
    )
"""

from typing import Optional, Dict, Any, List
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Prometheus metrics collector for system monitoring.

    Metrics:
    - agent_executions_total: Counter of agent executions
    - agent_latency_seconds: Histogram of agent latencies
    - llm_requests_total: Counter of LLM requests
    - llm_latency_seconds: Histogram of LLM latencies
    - llm_tokens_total: Histogram of token usage
    - tool_executions_total: Counter of tool executions
    - tool_latency_seconds: Histogram of tool latencies
    - concurrent_requests: Gauge of active requests
    - request_duration_seconds: Histogram of request durations
    """

    def __init__(self, enabled: bool = True, debug: bool = False):
        """
        Initialize metrics collector.

        Args:
            enabled: Enable/disable metrics
            debug: Enable debug logging
        """
        self.enabled = enabled
        self.debug = debug

        if not enabled:
            return

        try:
            # Initialize all metrics
            self._init_agent_metrics()
            self._init_llm_metrics()
            self._init_tool_metrics()
            self._init_request_metrics()
            self._init_rag_metrics()
            self._init_business_metrics()

            if debug:
                logger.info("Prometheus metrics initialized")

        except ImportError:
            logger.warning(
                "prometheus_client package not installed. "
                "Install: pip install prometheus-client"
            )
            self.enabled = False
        except Exception as e:
            logger.warning(f"Metrics initialization failed: {e}")
            self.enabled = False

    def _init_agent_metrics(self):
        """Initialize agent-related metrics."""
        # Agent execution counter
        self.agent_executions = Counter(
            "agent_executions_total",
            "Total agent executions",
            ["agent_name", "tenant_id", "status"]
        )

        # Agent latency histogram (with buckets for different latencies)
        self.agent_latency = Histogram(
            "agent_latency_seconds",
            "Agent execution latency in seconds",
            ["agent_name"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )

        # Agent confidence histogram
        self.agent_confidence = Histogram(
            "agent_confidence",
            "Agent confidence score",
            ["agent_name"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
        )

    def _init_llm_metrics(self):
        """Initialize LLM-related metrics."""
        # LLM request counter
        self.llm_requests = Counter(
            "llm_requests_total",
            "Total LLM requests",
            ["model", "provider", "status"]
        )

        # LLM latency histogram
        self.llm_latency = Histogram(
            "llm_latency_seconds",
            "LLM request latency in seconds",
            ["model", "provider"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )

        # LLM token usage
        self.llm_tokens = Histogram(
            "llm_tokens",
            "LLM token usage",
            ["model", "token_type"],  # token_type: input, output
            buckets=(10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000)
        )

        # LLM cost
        self.llm_cost = Counter(
            "llm_cost_usd_total",
            "Total LLM cost in USD",
            ["model"]
        )

    def _init_tool_metrics(self):
        """Initialize tool-related metrics."""
        # Tool execution counter
        self.tool_executions = Counter(
            "tool_executions_total",
            "Total tool executions",
            ["tool_name", "tenant_id", "status"]
        )

        # Tool latency histogram
        self.tool_latency = Histogram(
            "tool_latency_seconds",
            "Tool execution latency in seconds",
            ["tool_name"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )

    def _init_request_metrics(self):
        """Initialize request-related metrics."""
        # Concurrent requests gauge
        self.concurrent_requests = Gauge(
            "concurrent_requests",
            "Current number of concurrent requests"
        )

        # Request duration histogram
        self.request_duration = Histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            ["endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )

        # Request counter
        self.requests_total = Counter(
            "requests_total",
            "Total requests",
            ["endpoint", "status"]
        )

        # Chat-specific metrics
        self.chat_requests = Counter(
            "chat_requests_total",
            "Total chat requests",
            ["tenant_id", "session_id"]
        )

        self.chat_requests_completed = Counter(
            "chat_requests_completed_total",
            "Total completed chat requests",
            ["tenant_id", "session_id", "status"]
        )

        self.chat_request_latency = Histogram(
            "chat_request_latency_seconds",
            "Chat request latency in seconds",
            ["tenant_id"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        )

    def _init_rag_metrics(self):
        """Initialize RAG-related metrics."""
        # RAG retrieval duration
        self.rag_retrieval_latency = Histogram(
            "rag_retrieval_latency_seconds",
            "RAG retrieval latency in seconds",
            ["stage"],  # stage: stage_1, stage_2, reranker
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )

        # Documents retrieved
        self.rag_documents_retrieved = Histogram(
            "rag_documents_retrieved",
            "Number of documents retrieved",
            ["stage"],
            buckets=(1, 5, 10, 20, 50, 100, 200)
        )

    def _init_business_metrics(self):
        """Initialize business-related metrics."""
        # Intent classification counter
        self.intent_classifications = Counter(
            "intent_classifications_total",
            "Total intent classifications",
            ["intent", "tenant_id"]
        )

        # Sentiment classification counter
        self.sentiment_classifications = Counter(
            "sentiment_classifications_total",
            "Total sentiment classifications",
            ["sentiment", "tenant_id"]
        )

        # Escalations created
        self.escalations_created = Counter(
            "escalations_created_total",
            "Total escalations created",
            ["priority", "tenant_id", "channel"]
        )

    # ======================================================
    # AGENT METRICS
    # ======================================================

    def record_agent_execution(
        self,
        agent_name: str,
        tenant_id: str,
        status: str,
        latency_seconds: float,
        confidence: Optional[float] = None
    ):
        """
        Record agent execution metric.

        Args:
            agent_name: Name of the agent
            tenant_id: Tenant identifier
            status: Execution status (success, error, timeout)
            latency_seconds: Execution time in seconds
            confidence: Agent confidence score (optional)
        """
        if not self.enabled:
            return

        self.agent_executions.labels(
            agent_name=agent_name,
            tenant_id=tenant_id,
            status=status
        ).inc()

        self.agent_latency.labels(agent_name=agent_name).observe(latency_seconds)

        if confidence is not None:
            self.agent_confidence.labels(agent_name=agent_name).observe(confidence)

        if self.debug:
            logger.debug(
                f"Agent metric: {agent_name} ({tenant_id}) "
                f"status={status} latency={latency_seconds}s"
            )

    # ======================================================
    # LLM METRICS
    # ======================================================

    def record_llm_request(
        self,
        model: str,
        provider: str,
        status: str,
        latency_seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None
    ):
        """
        Record LLM request metric.

        Args:
            model: Model name
            provider: Provider name (groq, openrouter, etc.)
            status: Request status (success, error)
            latency_seconds: Request time in seconds
            input_tokens: Input token count
            output_tokens: Output token count
            cost_usd: Cost in USD (optional)
        """
        if not self.enabled:
            return

        self.llm_requests.labels(
            model=model,
            provider=provider,
            status=status
        ).inc()

        self.llm_latency.labels(model=model, provider=provider).observe(latency_seconds)

        self.llm_tokens.labels(model=model, token_type="input").observe(input_tokens)
        self.llm_tokens.labels(model=model, token_type="output").observe(output_tokens)

        if cost_usd is not None:
            self.llm_cost.labels(model=model).inc(cost_usd)

        if self.debug:
            logger.debug(
                f"LLM metric: {model} ({provider}) "
                f"tokens={input_tokens + output_tokens} latency={latency_seconds}s"
            )

    # ======================================================
    # TOOL METRICS
    # ======================================================

    def record_tool_execution(
        self,
        tool_name: str,
        tenant_id: str,
        status: str,
        latency_seconds: float
    ):
        """
        Record tool execution metric.

        Args:
            tool_name: Name of the tool
            tenant_id: Tenant identifier
            status: Execution status (success, error)
            latency_seconds: Execution time in seconds
        """
        if not self.enabled:
            return

        self.tool_executions.labels(
            tool_name=tool_name,
            tenant_id=tenant_id,
            status=status
        ).inc()

        self.tool_latency.labels(tool_name=tool_name).observe(latency_seconds)

        if self.debug:
            logger.debug(
                f"Tool metric: {tool_name} ({tenant_id}) "
                f"status={status} latency={latency_seconds}s"
            )

    # ======================================================
    # REQUEST METRICS
    # ======================================================

    def increment_concurrent_requests(self):
        """Increment concurrent requests gauge."""
        if self.enabled:
            self.concurrent_requests.inc()

    def decrement_concurrent_requests(self):
        """Decrement concurrent requests gauge."""
        if self.enabled:
            self.concurrent_requests.dec()

    def record_request(
        self,
        tenant_id: str,
        session_id: str
    ):
        """
        Record incoming request (counter).

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
        """
        if not self.enabled:
            return

        self.chat_requests.labels(
            tenant_id=tenant_id,
            session_id=session_id
        ).inc()

    def record_request_completion(
        self,
        tenant_id: str,
        session_id: str,
        status: str,
        latency_seconds: float
    ):
        """
        Record request completion with status and latency.

        Args:
            tenant_id: Tenant identifier
            session_id: Session identifier
            status: Request status (success, error)
            latency_seconds: Request duration in seconds
        """
        if not self.enabled:
            return

        self.chat_requests_completed.labels(
            tenant_id=tenant_id,
            session_id=session_id,
            status=status
        ).inc()

        self.chat_request_latency.labels(
            tenant_id=tenant_id
        ).observe(latency_seconds)

    # ======================================================
    # RAG METRICS
    # ======================================================

    def record_rag_retrieval(
        self,
        stage: str,
        latency_seconds: float,
        document_count: int
    ):
        """
        Record RAG retrieval metric.

        Args:
            stage: Retrieval stage (stage_1, stage_2, reranker)
            latency_seconds: Retrieval time in seconds
            document_count: Number of documents retrieved
        """
        if not self.enabled:
            return

        self.rag_retrieval_latency.labels(stage=stage).observe(latency_seconds)
        self.rag_documents_retrieved.labels(stage=stage).observe(document_count)

    # ======================================================
    # BUSINESS METRICS
    # ======================================================

    def record_intent_classification(
        self,
        intent: str,
        tenant_id: str
    ):
        """Record intent classification."""
        if self.enabled:
            self.intent_classifications.labels(intent=intent, tenant_id=tenant_id).inc()

    def record_sentiment_classification(
        self,
        sentiment: str,
        tenant_id: str
    ):
        """Record sentiment classification."""
        if self.enabled:
            self.sentiment_classifications.labels(sentiment=sentiment, tenant_id=tenant_id).inc()

    def record_escalation(
        self,
        priority: str,
        tenant_id: str,
        channel: str
    ):
        """Record escalation creation."""
        if self.enabled:
            self.escalations_created.labels(
                priority=priority,
                tenant_id=tenant_id,
                channel=channel
            ).inc()

    # ======================================================
    # METRICS EXPORT
    # ======================================================

    def generate_metrics(self) -> str:
        """
        Generate Prometheus metrics text format.

        Returns:
            Metrics in Prometheus text format
        """
        if not self.enabled:
            return ""

        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest()

    def get_content_type(self) -> str:
        """Get metrics content type for HTTP response."""
        from prometheus_client import CONTENT_TYPE_LATEST
        return CONTENT_TYPE_LATEST


# Metrics endpoint for FastAPI

def create_metrics_endpoint(metrics_collector: MetricsCollector):
    """
    Create FastAPI endpoint for metrics.

    Usage:
        from fastapi import FastAPI
        app = FastAPI()

        metrics = MetricsCollector()
        app.add_route("/metrics", create_metrics_endpoint(metrics))
    """
    from fastapi import Response
    from fastapi.responses import PlainTextResponse

    async def metrics_endpoint():
        return Response(
            content=metrics_collector.generate_metrics(),
            media_type=metrics_collector.get_content_type()
        )

    return metrics_endpoint
