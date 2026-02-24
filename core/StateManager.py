import copy
from typing import Dict, List, Optional
from models.patch import Patch


class StateManager:
    """Manages session-isolated state with version tracking and patch-based mutations.

    Core Architecture:
        - Session-per-isolation model: Each session has independent state
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
        """Initialize state manager with session-isolated storage.

        Creates:
            - _states: Dict mapping session_id to session state containers
            - Each session has its own snapshots, patches, and version counter
        """
        # Session-isolated storage: {session_id: {state, version, snapshots, patches}}
        self._sessions: Dict[str, Dict] = {}

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

    def _ensure_session(self, session_id: str) -> None:
        """Ensure session exists in storage. Initialize if not.

        Args:
            session_id: Unique session identifier
        """
        if session_id not in self._sessions:
            initial_state = self._initial_state()
            self._sessions[session_id] = {
                "state": initial_state,
                "version": 1,
                "snapshots": {1: initial_state},
                "patch_history": {}
            }

    # -----------------------------------------------------
    # PUBLIC ACCESSORS (SESSION-ISOLATED)
    # -----------------------------------------------------

    def get_session_version(self, session_id: str) -> int:
        """Get current version number for a session.

        Args:
            session_id: Session identifier

        Returns:
            int: Current version number for the session (starts at 1)

        Note:
            Creates session if it doesn't exist (lazy initialization)
        """
        self._ensure_session(session_id)
        return self._sessions[session_id]["version"]

    @property
    def current_version(self) -> int:
        """Get current version number for default session.

        DEPRECATED: Use get_session_version(session_id) instead.

        Returns:
            int: Current version number (for backward compatibility)
        """
        # For backward compatibility with code that doesn't pass session_id
        # This returns the highest version across all sessions
        if not self._sessions:
            return 1
        return max(session["version"] for session in self._sessions.values())

    @property
    def current_state(self) -> Dict:
        """Get current state as deep copy (prevents external mutation).

        DEPRECATED: Use get_session_state(session_id) instead.

        Returns:
            dict: Deep copy of current state at current_version

        Note:
            Returns the most recently updated session's state.
            This is for backward compatibility only.
        """
        if not self._sessions:
            return self._initial_state()

        # Find the session with the highest version (most recently active)
        latest_session_id = max(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid]["version"]
        )
        return self.get_session_state(latest_session_id)

    def get_session_state(self, session_id: str) -> Dict:
        """Get current state for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            dict: Deep copy of current state at current_version

        Note:
            Returns deep copy to ensure snapshot immutability.
            Creates session if it doesn't exist.
        """
        self._ensure_session(session_id)
        return copy.deepcopy(self._sessions[session_id]["state"])

    def get_snapshot(self, version: int, session_id: Optional[str] = None) -> Dict:
        """Get state snapshot at specific version.

        Args:
            version: Version number to retrieve
            session_id: Session identifier (optional, for multi-session)

        Returns:
            dict: Deep copy of state at requested version

        Raises:
            ValueError: If requested version does not exist in snapshot history
        """
        if session_id:
            self._ensure_session(session_id)
            snapshots = self._sessions[session_id]["snapshots"]
        else:
            # Backward compatibility: search all sessions
            for session_data in self._sessions.values():
                if version in session_data["snapshots"]:
                    return copy.deepcopy(session_data["snapshots"][version])
            raise ValueError("Snapshot version does not exist")

        if version not in snapshots:
            raise ValueError(f"Snapshot version {version} does not exist for session {session_id}")
        return copy.deepcopy(snapshots[version])

    def get_patch(self, version: int, session_id: Optional[str] = None) -> Patch:
        """Get patch that created specific version.

        Args:
            version: Version number whose patch to retrieve
            session_id: Session identifier (optional)

        Returns:
            Patch: The patch object that created this version

        Raises:
            ValueError: If requested version does not exist in patch history
        """
        if session_id:
            self._ensure_session(session_id)
            patch_history = self._sessions[session_id]["patch_history"]
        else:
            # Backward compatibility: search all sessions
            for session_data in self._sessions.values():
                if version in session_data["patch_history"]:
                    return session_data["patch_history"][version]
            raise ValueError("Patch version does not exist")

        if version not in patch_history:
            raise ValueError(f"Patch version {version} does not exist for session {session_id}")
        return patch_history[version]

    # -----------------------------------------------------
    # PATCH APPLICATION (CORE ENGINE)
    # -----------------------------------------------------

    def apply_patch(self, patch: Patch, session_id: Optional[str] = None) -> int:
        """Apply validated patch and create new version snapshot.

        This is the ONLY method that modifies state. Patches must be
        validated by PatchValidator before being applied.

        Args:
            patch: Validated Patch object
            session_id: Session identifier (optional, for multi-session)

        Returns:
            int: New version number after patch application

        Enforces:
            - Section replacement (entire section replaced, not merged)
            - Immutable snapshot (new version, old versions unchanged)
            - Version increment (monotonically increasing per session)
            - Patch history (append-only recording)

        Process:
            1. Deep copy current state (base)
            2. Replace target section with patch.changes
            3. Increment version counter
            4. Persist new snapshot
            5. Record patch in history
            6. Update current pointer
        """
        # Use session_id from metadata if not provided
        if session_id is None:
            session_id = patch.metadata.get("session_id", "default")

        self._ensure_session(session_id)
        session = self._sessions[session_id]

        # Step 1: Get current state (immutable copy)
        base_state = copy.deepcopy(session["state"])

        # Step 2: Replace entire section
        target_section = patch.target_section
        base_state[target_section] = patch.changes

        # Step 3: Increment version
        new_version = session["version"] + 1

        # Step 4: Persist snapshot
        session["snapshots"][new_version] = base_state

        # Step 5: Persist patch
        session["patch_history"][new_version] = patch

        # Step 6: Update state and version
        session["state"] = base_state
        session["version"] = new_version

        return new_version

    # -----------------------------------------------------
    # SESSION MANAGEMENT
    # -----------------------------------------------------

    def create_session(self, session_id: str) -> Dict:
        """Create a new isolated session.

        Args:
            session_id: Unique session identifier

        Returns:
            dict: Initial state for the new session

        Note:
            If session already exists, returns its current state.
        """
        self._ensure_session(session_id)
        return self.get_session_state(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all its history.

        Args:
            session_id: Session identifier to delete

        Note:
            Use for cleanup when sessions expire.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """List all active session IDs.

        Returns:
            List of session identifiers
        """
        return list(self._sessions.keys())

    # -----------------------------------------------------
    # REPLAY SUPPORT
    # -----------------------------------------------------

    def replay_from_scratch(self, session_id: Optional[str] = None) -> Dict:
        """Rebuild state deterministically from version 1.

        Args:
            session_id: Session to replay (optional)

        Returns:
            dict: Reconstructed state by applying all patches in order

        Useful for:
            - Integrity validation
            - Debugging state evolution
            - Reproducing specific versions
            - Testing patch application logic
        """
        state = self._initial_state()

        if session_id:
            self._ensure_session(session_id)
            patch_history = self._sessions[session_id]["patch_history"]
        else:
            # Replay from first session (backward compatibility)
            if not self._sessions:
                return state
            first_session = next(iter(self._sessions.values()))
            patch_history = first_session["patch_history"]

        for version in sorted(patch_history.keys()):
            patch = patch_history[version]
            state[patch.target_section] = patch.changes

        return state

    # -----------------------------------------------------
    # SAFETY CHECKS
    # -----------------------------------------------------

    def validate_integrity(self, session_id: Optional[str] = None) -> bool:
        """Validate state integrity by replaying all patches.

        Args:
            session_id: Session to validate (optional)

        Returns:
            bool: True if replayed state matches current snapshot (integrity OK)

        Use case:
            Run after critical operations or on schedule to detect
            data corruption or inconsistencies.
        """
        if session_id:
            if session_id not in self._sessions:
                return True
            replayed = self.replay_from_scratch(session_id)
            return replayed == self._sessions[session_id]["state"]
        else:
            # Validate all sessions
            return all(
                self.replay_from_scratch(sid) == self._sessions[sid]["state"]
                for sid in self._sessions
            )