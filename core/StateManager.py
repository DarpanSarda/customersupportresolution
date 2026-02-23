import copy
from typing import Dict, List
from models.patch import Patch


class StateManager:
    """Manages global state with version tracking and patch-based mutations.

    Core Architecture:
        - Snapshot-per-version model: Each version is immutable
        - Patch-based mutation: All changes via validated patches
        - One-section-per-patch: Each patch modifies exactly one section
        - Version increment: Every successful patch creates new version
        - Append-only history: Patches and snapshots never deleted

    State Sections (top-level keys - FROZEN):
        - meta: Metadata (tenant_id, session_id, timestamps)
        - lifecycle: Execution status (phase, step, active_agents)
        - conversation: User messages, responses, history
        - context: Current execution context (user_profile, channel)
        - understanding: Intent, sentiment, entities (Tier 1 output)
        - decision: Policy decisions, reasoning (Tier 2 output)
        - execution: Tool execution results, action statuses
        - memory: Long-term conversation memory, summaries
        - async_registry: Async tool tracking (pending operations)
        - audit: Append-only audit log (all events, all patches)
    """

    def __init__(self):
        """Initialize state manager with empty initial state at version 1.

        Creates:
            - Version counter starting at 1
            - Empty snapshot history dict
            - Empty patch history dict
            - Initial empty state snapshot at version 1
        """
        self._current_version: int = 1
        self._snapshots: Dict[int, Dict] = {}
        self._patch_history: Dict[int, Patch] = {}

        # Initialize first snapshot
        initial_state = self._initial_state()
        self._snapshots[self._current_version] = initial_state

    # -----------------------------------------------------
    # INITIAL STATE ENVELOPE (FROZEN STRUCTURE)
    # -----------------------------------------------------

    def _initial_state(self) -> Dict:
        """Create initial empty state structure.

        Returns:
            dict: Empty state with all top-level sections initialized as empty dicts

        Note:
            Top-level keys are frozen. New sections cannot be added.
        """
        return {
            "meta": {},
            "lifecycle": {},
            "conversation": {},
            "context": {},
            "understanding": {},
            "decision": {},
            "execution": {},
            "memory": {},
            "async_registry": {},
            "audit": {}
        }

    # -----------------------------------------------------
    # PUBLIC ACCESSORS
    # -----------------------------------------------------

    @property
    def current_version(self) -> int:
        """Get current version number.

        Returns:
            int: Current version number (starts at 1, increments per patch)
        """
        return self._current_version

    @property
    def current_state(self) -> Dict:
        """Get current state as deep copy (prevents external mutation).

        Returns:
            dict: Deep copy of current state at current_version

        Note:
            Returns deep copy to ensure snapshot immutability.
            External changes to returned dict do not affect stored state.
        """
        # Return deep copy to prevent mutation
        return copy.deepcopy(self._snapshots[self._current_version])

    def get_snapshot(self, version: int) -> Dict:
        """Get state snapshot at specific version.

        Args:
            version: Version number to retrieve

        Returns:
            dict: Deep copy of state at requested version

        Raises:
            ValueError: If requested version does not exist in snapshot history
        """
        if version not in self._snapshots:
            raise ValueError("Snapshot version does not exist")
        return copy.deepcopy(self._snapshots[version])

    def get_patch(self, version: int) -> Patch:
        """Get patch that created specific version.

        Args:
            version: Version number whose patch to retrieve

        Returns:
            Patch: The patch object that created this version

        Raises:
            ValueError: If requested version does not exist in patch history
        """
        if version not in self._patch_history:
            raise ValueError("Patch version does not exist")
        return self._patch_history[version]

    # -----------------------------------------------------
    # PATCH APPLICATION (CORE ENGINE)
    # -----------------------------------------------------

    def apply_patch(self, patch: Patch) -> int:
        """Apply validated patch and create new version snapshot.

        This is the ONLY method that modifies state. Patches must be
        validated by PatchValidator before being applied.

        Args:
            patch: Validated Patch object containing:
                - patch_id: Unique identifier
                - agent_name: Agent creating the patch
                - target_section: Section to modify
                - confidence: Agent confidence (0-1)
                - changes: Dictionary of changes to apply
                - metadata: Execution metadata

        Returns:
            int: New version number after patch application

        Enforces:
            - Section replacement (entire section replaced, not merged)
            - Immutable snapshot (new version, old versions unchanged)
            - Version increment (monotonically increasing)
            - Patch history (append-only recording)

        Process:
            1. Deep copy current state (base)
            2. Replace target section with patch.changes
            3. Increment version counter
            4. Persist new snapshot
            5. Record patch in history
            6. Update current pointer
        """
        # Step 1: Get current state (immutable copy)
        base_state = self.current_state

        # Step 2: Replace entire section
        target_section = patch.target_section
        base_state[target_section] = patch.changes

        # Step 3: Increment version
        new_version = self._current_version + 1

        # Step 4: Persist snapshot
        self._snapshots[new_version] = base_state

        # Step 5: Persist patch
        self._patch_history[new_version] = patch

        # Step 6: Move pointer
        self._current_version = new_version

        return new_version

    # -----------------------------------------------------
    # REPLAY SUPPORT
    # -----------------------------------------------------

    def replay_from_scratch(self) -> Dict:
        """Rebuild state deterministically from version 1.

        Useful for:
            - Integrity validation
            - Debugging state evolution
            - Reproducing specific versions
            - Testing patch application logic

        Returns:
            dict: Reconstructed state by applying all patches in order
        """
        state = self._initial_state()

        for version in sorted(self._patch_history.keys()):
            patch = self._patch_history[version]
            state[patch.target_section] = patch.changes

        return state

    # -----------------------------------------------------
    # SAFETY CHECKS
    # -----------------------------------------------------

    def validate_integrity(self) -> bool:
        """Validate state integrity by replaying all patches.

        Confirms that replaying patches from scratch produces
        the same state as the current snapshot.

        Returns:
            bool: True if replayed state matches current snapshot (integrity OK)

        Use case:
            Run after critical operations or on schedule to detect
            data corruption or inconsistencies.
        """
        replayed = self.replay_from_scratch()
        return replayed == self._snapshots[self._current_version]