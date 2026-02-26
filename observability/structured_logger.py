"""
Structured JSON logger for consistent logging across the system.

Provides:
- JSON formatted log output
- Correlation ID injection (trace_id, tenant_id, session_id)
- Structured event logging
- Error logging with stack traces

Usage:
    logger = StructuredLogger("customer-support-resolution", level="INFO")
    logger.info({"event": "agent_execution", "agent": "IntentAgent"})
    logger.error({"event": "tool_failed", "tool": "faq_lookup"}, exception=e)
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime


class StructuredLogger:
    """
    Structured JSON logger for enterprise logging.

    Logs are output as JSON with consistent schema:
    {
        "timestamp": "2024-02-25T00:00:00Z",
        "level": "INFO",
        "service": "customer-support-resolution",
        "event": "agent_execution",
        "trace_id": "...",
        "tenant_id": "...",
        "session_id": "...",
        ... additional event fields
    }
    """

    def __init__(self, service_name: str, level: str = "INFO", json_output: bool = True):
        """
        Initialize structured logger.

        Args:
            service_name: Name of the service for log attribution
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            json_output: Enable JSON output (False for plain text)
        """
        self.service_name = service_name
        self.json_output = json_output

        # Configure logging
        self._logger = logging.getLogger(service_name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Clear existing handlers
        self._logger.handlers.clear()

        # Create handler
        handler = logging.StreamHandler(sys.stdout)

        # Set formatter
        if json_output:
            handler.setFormatter(JSONFormatter(service_name))
        else:
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )

        self._logger.addHandler(handler)

    def info(self, event: Dict[str, Any], **context) -> None:
        """
        Log info level event.

        Args:
            event: Event data dictionary
            **context: Additional context (trace_id, tenant_id, session_id)

        Example:
            logger.info(
                {"event": "agent_execution", "agent": "IntentAgent", "confidence": 0.95},
                trace_id="abc-123",
                tenant_id="acme"
            )
        """
        log_data = self._prepare_log_data(event, context)
        self._logger.info(log_data)

    def warning(self, event: Dict[str, Any], **context) -> None:
        """
        Log warning level event.

        Args:
            event: Event data dictionary
            **context: Additional context
        """
        log_data = self._prepare_log_data(event, context)
        self._logger.warning(log_data)

    def error(
        self,
        event: Dict[str, Any],
        exception: Optional[Exception] = None,
        **context
    ) -> None:
        """
        Log error level event.

        Args:
            event: Event data dictionary
            exception: Exception object for stack trace
            **context: Additional context

        Example:
            logger.error(
                {"event": "tool_failed", "tool": "faq_lookup"},
                exception=e,
                trace_id="abc-123"
            )
        """
        log_data = self._prepare_log_data(event, context)

        if exception:
            import traceback
            log_data["exception_type"] = type(exception).__name__
            log_data["exception_message"] = str(exception)
            log_data["stack_trace"] = traceback.format_exc()

        self._logger.error(log_data)

    def debug(self, event: Dict[str, Any], **context) -> None:
        """
        Log debug level event.

        Args:
            event: Event data dictionary
            **context: Additional context
        """
        log_data = self._prepare_log_data(event, context)
        self._logger.debug(log_data)

    def _prepare_log_data(self, event: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare log data with service info and context.

        Args:
            event: Event data
            context: Additional context

        Returns:
            Complete log data dictionary
        """
        log_data = {
            "service": self.service_name,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Add correlation IDs from context
        correlation_fields = ["trace_id", "request_id", "tenant_id", "session_id"]
        for field in correlation_fields:
            if field in context:
                log_data[field] = context[field]

        # Add event data
        log_data.update(event)

        return log_data

    @property
    def level(self) -> int:
        """Get current log level."""
        return self._logger.level

    def set_level(self, level: str) -> None:
        """
        Set log level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """

    def __init__(self, service_name: str):
        """
        Initialize JSON formatter.

        Args:
            service_name: Service name for log attribution
        """
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON formatted log string
        """
        import json

        # Extract log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception_type"] = record.exc_info[0].__name__
            log_data["exception_message"] = str(record.exc_info[1])
            log_data["stack_trace"] = self.formatException(record.exc_info)

        # Add log message or data
        if isinstance(record.msg, dict):
            log_data.update(record.msg)
        else:
            log_data["message"] = record.getMessage()

        return json.dumps(log_data)
