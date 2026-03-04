"""OpenRouter LLM client implementation."""

import os
import time
from openai import OpenAI
from typing import Dict, List, Optional, Any
from models.llm import LLMResponse
from llms.LLMManager import LLMManager
from dotenv import load_dotenv

load_dotenv()


class OpenRouterClient(LLMManager):
    """OpenRouter API client for LLM generation with Langfuse observability."""

    # OpenRouter model pricing (per 1M tokens)
    MODEL_PRICING = {
        "openai/gpt-oss-20b": {"input": 0.10, "output": 0.10},
        "anthropic/claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3.5-sonnet:beta": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3-opus": {"input": 15.0, "output": 75.0},
        "anthropic/claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai/gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "google/gemini-pro-1.5": {"input": 1.25, "output": 5.0},
        "meta-llama/llama-3.1-70b-instruct": {"input": 0.59, "output": 0.79},
        "meta-llama/llama-3.1-8b-instruct": {"input": 0.07, "output": 0.07},
    }

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        langfuse_tracer=None
    ):
        """Initialize OpenRouter client.

        Args:
            model: Model name to use
            langfuse_tracer: LangfuseTracer instance for observability
        """
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
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
        # Try exact match first
        if self.model in self.MODEL_PRICING:
            return self.MODEL_PRICING[self.model]

        # Try prefix match for model variants
        for model_name, pricing in self.MODEL_PRICING.items():
            if self.model.startswith(model_name.split(":")[0]):
                return pricing

        # Default pricing
        return {"input": 1.0, "output": 1.0}

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
        """Generate completion from OpenRouter.

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
                agent_name=agent_name or "OpenRouterClient",
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
