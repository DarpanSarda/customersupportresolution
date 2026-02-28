# agents/context_builder_agent.py

import json
import re
from core.BaseAgent import BaseAgent, AgentExecutionContext
from models.patch import Patch
from typing import Dict, Any, List, Optional


class ContextBuilderAgent(BaseAgent):
    """
    LLM-based structured entity extraction agent.

    Extracts entities from user messages based on tenant-specific schemas.
    Generic, tenant-aware, config-driven, with strict output validation.

    Reads:
    - understanding.input.raw_text: User message to extract from
    - context.tenant_id: Tenant ID for loading correct schema

    Writes:
    - context.entities: Extracted and validated entities
    - context.extraction_confidence: Confidence in extraction

    Key design:
    - Entity schemas come from config (tenant-specific)
    - LLM extracts ONLY defined fields
    - Strict validation removes unknown fields
    - Type casting and null-safety enforced
    """

    agent_name = "ContextBuilderAgent"
    allowed_section = "context"

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """
        Extract entities from user message using tenant-specific schema.

        Returns Patch with extracted entities.
        """
        # -------------------------------------------------
        # 1️⃣ Extract inputs from state
        # -------------------------------------------------
        understanding = state.get("understanding", {})
        input_data = understanding.get("input", {})
        raw_text = input_data.get("raw_text", "")

        if not raw_text:
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={"entities": {}, "extraction_confidence": 0.0}
            )

        # Get tenant ID
        context_data = state.get("context", {})
        tenant_id = context_data.get("tenant_id") or (
            context.tenant_id if context else "default"
        )

        # -------------------------------------------------
        # 2️⃣ Load tenant entity schema
        # -------------------------------------------------
        entity_schema = self.config_loader.get_entity_schema(tenant_id)

        if not entity_schema:
            # No schema defined for tenant - return empty
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "entities": {},
                    "extraction_confidence": 0.0,
                    "extraction_reason": f"No entity schema defined for tenant '{tenant_id}'"
                }
            )

        # -------------------------------------------------
        # 3️⃣ Build extraction prompt
        # -------------------------------------------------
        prompt = self._build_extraction_prompt(raw_text, entity_schema)

        # -------------------------------------------------
        # 4️⃣ Call LLM with retry
        # -------------------------------------------------
        llm_output = self._extract_with_retry(prompt, max_retries=2)

        if not llm_output:
            # Extraction failed
            return Patch(
                agent_name=self.agent_name,
                target_section=self.allowed_section,
                confidence=1.0,
                changes={
                    "entities": {},
                    "extraction_confidence": 0.0,
                    "extraction_reason": "LLM extraction failed after retries"
                }
            )

        # -------------------------------------------------
        # 5️⃣ Validate and filter entities
        # -------------------------------------------------
        validated_entities = self._validate_and_filter(
            llm_output,
            entity_schema
        )

        # Merge with existing entities (accumulative)
        existing_entities = context_data.get("entities", {})
        merged_entities = {**existing_entities, **validated_entities}

        # Calculate confidence based on extraction completeness
        total_fields = len(entity_schema.get("fields", {}))
        extracted_fields = len([v for v in validated_entities.values() if v is not None])
        confidence = extracted_fields / total_fields if total_fields > 0 else 0.0

        # -------------------------------------------------
        # 6️⃣ Return Patch
        # -------------------------------------------------
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=confidence,
            changes={
                "tenant_id": tenant_id,
                "entities": merged_entities,
                "extraction_confidence": confidence,
                "extracted_fields": list(validated_entities.keys()),
                "extraction_reason": f"Extracted {extracted_fields}/{total_fields} fields"
            }
        )

    def _build_extraction_prompt(self, raw_text: str, entity_schema: dict) -> str:
        """
        Build strict JSON extraction prompt based on entity schema.
        """
        fields_desc = entity_schema.get("fields", {})
        description = entity_schema.get("description", "Entity extraction")

        # Build field list with types
        field_list = []
        for field_name, field_config in fields_desc.items():
            field_type = field_config.get("type", "string")
            field_desc = field_config.get("description", "")
            field_list.append(f"- {field_name} ({field_type}): {field_desc}")

        fields_str = "\n".join(field_list)

        prompt = f"""You are a precise entity extraction system.

Your task: Extract ONLY the specified fields from the user message.

{description}

Allowed fields (extract ONLY these):
{fields_str}

Rules:
- Return ONLY valid JSON
- If a field value is not found, use null
- Do NOT include any fields not listed above
- Do NOT include explanations or extra text
- Extract numbers as numbers (not strings)
- Keep text values as strings

User message:
{raw_text}

Return JSON in this exact format:
{{{{"field_name": value or null, ...}}}}"""

        return prompt

    def _extract_with_retry(self, prompt: str, max_retries: int = 2) -> Optional[dict]:
        """
        Call LLM with JSON extraction and retry logic.
        """
        for attempt in range(max_retries + 1):
            try:
                messages = [{"role": "user", "content": prompt}]
                llm_response = self.llm_client.generate(messages)

                # Extract JSON from response
                content = llm_response.content.strip()

                # Try to find JSON in response (handle markdown code blocks)
                json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                # Handle nested braces for larger JSON
                if content.count('{') > content.count('}'):
                    # Find the complete JSON by balancing braces
                    brace_count = 0
                    json_start = content.find('{')
                    json_end = json_start
                    for i, char in enumerate(content[json_start:], json_start):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                    content = content[json_start:json_end]

                parsed = json.loads(content)
                return parsed

            except (json.JSONDecodeError, Exception) as e:
                if attempt < max_retries:
                    continue
                # Last attempt failed
                return None

        return None

    def _validate_and_filter(self, extracted: dict, entity_schema: dict) -> dict:
        """
        Validate extracted entities against schema.
        - Remove unknown fields
        - Cast types
        - Null safety
        """
        validated = {}
        fields_config = entity_schema.get("fields", {})

        for field_name, field_config in fields_config.items():
            field_type = field_config.get("type", "string")
            raw_value = extracted.get(field_name)

            # Handle null/missing
            if raw_value is None:
                validated[field_name] = None
                continue

            # Type casting
            try:
                if field_type == "number":
                    # Handle numeric strings and actual numbers
                    if isinstance(raw_value, str):
                        # Remove currency symbols and commas
                        cleaned = re.sub(r'[^\d.-]', '', raw_value)
                        validated[field_name] = float(cleaned) if cleaned else None
                    else:
                        validated[field_name] = float(raw_value)
                elif field_type == "string":
                    validated[field_name] = str(raw_value).strip()
                elif field_type == "boolean":
                    if isinstance(raw_value, bool):
                        validated[field_name] = raw_value
                    elif isinstance(raw_value, str):
                        validated[field_name] = raw_value.lower() in ("true", "yes", "1")
                    else:
                        validated[field_name] = bool(raw_value)
                else:
                    validated[field_name] = raw_value
            except (ValueError, TypeError):
                validated[field_name] = None

        return validated
