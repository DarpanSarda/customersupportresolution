"""Groq LLM client implementation."""

import os
import time
from groq import Groq
from typing import Dict, List, Optional, Any
from models.llm import LLMResponse
from llms.LLMManager import LLMManager
from dotenv import load_dotenv

load_dotenv()


class GroqClient(LLMManager):
    """Groq API client for LLM generation with Langfuse observability."""

    # Groq model pricing (per 1M tokens)
    MODEL_PRICING = {
        "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},  # Free on Groq
        "llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},
        "llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},
        "mixtral-8x7b-32768": {"input": 0.0, "output": 0.0},
        "gemma2-9b-it": {"input": 0.0, "output": 0.0},
    }

    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        langfuse_tracer=None
    ):
        """Initialize Groq client.

        Args:
            model: Model name to use
            langfuse_tracer: LangfuseTracer instance for observability
        """
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self._langfuse = langfuse_tracer

    def _extract_prompt_content(self, messages: List[Dict]) -> str:
        """Extract prompt content from messages for observability.

        Args:
            messages: List of message dicts

        Returns:
            Concatenated prompt string
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    def _get_pricing(self) -> Dict[str, float]:
        """Get pricing for current model.

        Returns:
            Dict with input/output pricing per 1M tokens
        """
        return self.MODEL_PRICING.get(
            self.model,
            {"input": 0.0, "output": 0.0}  # Default to free
        )

    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        agent_name: Optional[str] = None,
        trace_id: Optional[str] = None,
        observation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate completion from Groq.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            agent_name: Name of the agent calling (for tracing)
            trace_id: Langfuse trace ID
            observation_id: Parent observation ID (for nested calls)
            metadata: Additional metadata for tracing

        Returns:
            LLMResponse with content, model, usage, and raw response
        """
        pricing = self._get_pricing()

        # Langfuse observation tracking
        langfuse_observation = None
        if self._langfuse and self._langfuse.enabled:
            langfuse_observation = self._langfuse.create_observation(
                trace_id=trace_id or self._langfuse.generate_trace_id(),
                agent_name=agent_name or "GroqClient",
                input_messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                observation_id=observation_id,
                metadata=metadata or {}
            )

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            latency_ms = int((time.time() - start_time) * 1000)

            usage = response.usage.model_dump() if response.usage else {}
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage=usage,
                raw=response.model_dump() if hasattr(response, 'model_dump') else None,
                latency_ms=latency_ms
            )

            # Finalize Langfuse observation
            if langfuse_observation:
                langfuse_observation.finalize(
                    output=response.choices[0].message.content,
                    usage=usage,
                    latency_ms=latency_ms,
                    pricing=pricing
                )

            return llm_response

        except Exception as e:
            # Finalize Langfuse observation with error
            if langfuse_observation:
                langfuse_observation.finalize_error(
                    error_message=str(e),
                    error_type=type(e).__name__
                )
            raise
