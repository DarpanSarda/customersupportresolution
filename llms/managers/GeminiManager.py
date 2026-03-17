"""
GeminiManager - LLM manager for Google Gemini API.

Implements BaseLLM interface for Google's Gemini models (Gemini 2.0, 1.5, etc.).
Self-registers with LLMFactory using @register_manager decorator.
"""

import asyncio
from typing import Dict, List, Optional, Any, AsyncIterator
from llms.BaseLLM import BaseLLM, LLMConfig, LLMResponse
from llms.LLMFactory import register_manager


@register_manager("gemini")
class GeminiManager(BaseLLM):
    """
    Google Gemini LLM manager.

    Usage:
        config = LLMConfig(
            provider="gemini",
            api_key="AIza...",
            model="gemini-2.0-flash-exp"
        )
        llm = GeminiManager(config)
        response = await llm.generate("Hello!")
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._initialized = False

    def _validate_config(self) -> None:
        """Validate Gemini configuration."""
        if not self.config.api_key:
            raise ValueError("Google API key is required")

    async def _initialize_client(self):
        """Lazy initialize the GenerativeModel client."""
        if self._initialized:
            return

        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package is required. Install with: pip install google-generativeai"
            )

        # Configure the API key
        import google.generativeai as genai
        genai.configure(api_key=self.config.api_key)

        # Create the model
        self._genai = genai
        self._model = genai.GenerativeModel(self.config.model)
        self._initialized = True

    def _convert_temperature(self, temp: float) -> float:
        """Convert temperature to Gemini's 0-2 range if needed."""
        # Gemini uses 0-2 range, OpenAI uses 0-1
        # If temp is in OpenAI range, convert it
        if temp <= 1.0:
            return temp * 2.0
        return temp

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from a single prompt string."""
        await self._initialize_client()

        # Build generation config
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = self._convert_temperature(temperature)
        else:
            gen_config["temperature"] = self._convert_temperature(self.config.temperature)

        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        elif self.config.max_tokens:
            gen_config["max_output_tokens"] = self.config.max_tokens

        if self.config.top_p:
            gen_config["top_p"] = self.config.top_p

        # Add any extra params
        gen_config.update(self.config.extra_params or {})
        gen_config.update(kwargs)

        try:
            # Run in thread pool since Gemini's SDK is synchronous
            result = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config=self._genai.types.GenerationConfig(**gen_config)
            )

            # Extract metadata
            usage = result.usage_metadata if hasattr(result, 'usage_metadata') else None

            return LLMResponse(
                content=result.text,
                model=self.config.model,
                provider="gemini",
                tokens_used=usage.total_token_count if usage else None,
                finish_reason=result.candidates[0].finish_reason.name if result.candidates else None,
                raw_response={
                    "prompt_token_count": usage.prompt_token_count if usage else None,
                    "candidates_token_count": usage.candidates_token_count if usage else None,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}") from e

    async def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from chat messages."""
        await self._initialize_client()

        # Convert messages to Gemini's chat history format
        # Gemini uses a "conversation" approach with roles: "user" and "model"
        chat_history = []

        # Skip the last message (it's the current prompt)
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            chat_history.append({
                "role": role,
                "parts": [msg["content"]]
            })

        # Start a chat session
        chat = self._model.start_chat(history=chat_history)

        # Build generation config
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = self._convert_temperature(temperature)
        else:
            gen_config["temperature"] = self._convert_temperature(self.config.temperature)

        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        elif self.config.max_tokens:
            gen_config["max_output_tokens"] = self.config.max_tokens

        if self.config.top_p:
            gen_config["top_p"] = self.config.top_p

        gen_config.update(self.config.extra_params or {})
        gen_config.update(kwargs)

        try:
            # Get the last message (current prompt)
            current_prompt = messages[-1]["content"]

            # Send message and get response
            result = await asyncio.to_thread(
                chat.send_message,
                current_prompt,
                generation_config=self._genai.types.GenerationConfig(**gen_config)
            )

            usage = result.usage_metadata if hasattr(result, 'usage_metadata') else None

            return LLMResponse(
                content=result.text,
                model=self.config.model,
                provider="gemini",
                tokens_used=usage.total_token_count if usage else None,
                finish_reason=result.candidates[0].finish_reason.name if result.candidates else None,
                raw_response={
                    "prompt_token_count": usage.prompt_token_count if usage else None,
                    "candidates_token_count": usage.candidates_token_count if usage else None,
                }
            )

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}") from e

    async def stream_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a response from a single prompt string."""
        await self._initialize_client()

        # Build generation config
        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = self._convert_temperature(temperature)
        else:
            gen_config["temperature"] = self._convert_temperature(self.config.temperature)

        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        elif self.config.max_tokens:
            gen_config["max_output_tokens"] = self.config.max_tokens

        if self.config.top_p:
            gen_config["top_p"] = self.config.top_p

        gen_config.update(self.config.extra_params or {})
        gen_config.update(kwargs)

        try:
            # Generate content with streaming
            result = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                generation_config=self._genai.types.GenerationConfig(**gen_config),
                stream=True
            )

            # Iterate through chunks
            for chunk in result:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            raise RuntimeError(f"Gemini streaming error: {str(e)}") from e

    async def close(self):
        """Clean up resources."""
        if self._model:
            self._model = None
            self._initialized = False
