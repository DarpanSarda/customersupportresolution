from typing import Dict, Type
from models.patch import Patch
from pydantic import ValidationError


class PatchValidator:
    """Validates agent-produced patches before merging into state.

    Enforces multi-tenant isolation and section ownership rules.

    Validation Checks:
        1. Agent Registration: Agent must be in registry
        2. Section Authorization: Agent can only write to its assigned section
        3. Section Exists: Target section must exist in state
        4. Schema Validation: Changes must match section schema
        5. Cross-Section Safety: No top-level state keys in changes
    """

    def __init__(
        self,
        agent_registry: Dict[str, str],   # agent_name → allowed_section
        section_schemas: Dict[str, Type]
    ):
        """Initialize validator with agent registry and section schemas.

        Args:
            agent_registry: Mapping of agent names to their allowed sections.
                Example: {"intent-classifier-v1": "understanding"}

            section_schemas: Mapping of section names to Pydantic models.
                Example: {"understanding": UnderstandingSectionSchema}
        """
        self.agent_registry = agent_registry
        self.section_schemas = section_schemas

    def validate(self, patch: Patch, current_state: Dict) -> bool:
        """Validate a patch against all security and schema rules.

        Args:
            patch: Patch object containing:
                - patch_id: Unique identifier
                - agent_name: Agent that created the patch
                - target_section: Section to modify
                - confidence: Agent confidence (0-1)
                - changes: Dictionary of changes to apply
                - metadata: Execution metadata

            current_state: Current global state dictionary with top-level sections

        Returns:
            bool: True if all validation passes

        Raises:
            ValueError: If validation fails with specific reason:
                - "Unregistered agent": Agent not in registry
                - "{agent} not allowed to write to {section}": Section authorization failed
                - "Invalid section {section}": Section doesn't exist in state
                - "No schema defined for section {section}": Schema missing
                - "Section schema validation failed": Pydantic validation error
                - "Patch attempting cross-section mutation": Changes contain top-level keys
        """
        # -------------------------------------------------
        # 1️⃣ Agent Registration
        # -------------------------------------------------
        if patch.agent_name not in self.agent_registry:
            raise ValueError(
                f"Unregistered agent: {patch.agent_name}"
            )

        # -------------------------------------------------
        # 2️⃣ Section Authorization
        # -------------------------------------------------
        # System agents (prefixed with "System") can write to any section
        # This allows initialization of context, lifecycle, etc. before agents run
        is_system_agent = patch.agent_name.startswith("System")

        if not is_system_agent:
            allowed_section = self.agent_registry[patch.agent_name]

            if patch.target_section != allowed_section:
                raise ValueError(
                    f"{patch.agent_name} not allowed to write to {patch.target_section}"
                )

        # -------------------------------------------------
        # 3️⃣ Section Exists in State
        # -------------------------------------------------
        if patch.target_section not in current_state:
            raise ValueError(
                f"Invalid section: {patch.target_section}"
            )

        # -------------------------------------------------
        # 4️⃣ Section Schema Validation
        # -------------------------------------------------
        if patch.target_section not in self.section_schemas:
            raise ValueError(
                f"No schema defined for section {patch.target_section}"
            )

        schema_model = self.section_schemas[patch.target_section]

        try:
            schema_model(**patch.changes)
        except ValidationError as e:
            raise ValueError(
                f"Section schema validation failed: {str(e)}"
            )

        # -------------------------------------------------
        # 5️⃣ Cross-Section Safety Check
        # -------------------------------------------------
        # Ensure no top-level fields appear inside changes
        if any(key in current_state for key in patch.changes.keys()):
            raise ValueError(
                "Patch attempting cross-section mutation"
            )

        return True