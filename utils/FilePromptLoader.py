"""
FilePromptLoader - Load prompts from file system.

Prompt file structure:
    prompts/
    ├── IntentAgent/
    │   ├── v1.txt
    │   └── v2.txt
    ├── SentimentAgent/
    │   ├── v1.txt
    │   └── v2.txt
    └── ...
"""

import os
from pathlib import Path
from typing import Optional


class FilePromptLoader:
    """
    Load prompts from file system.

    Prompts are organized as: prompts/{agent_name}/{version}.txt
    """

    def __init__(self, prompts_dir: str = "prompts"):
        """
        Initialize FilePromptLoader.

        Args:
            prompts_dir: Directory containing prompt files (default: "prompts")
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[tuple[str, str], str] = {}

    def get_prompt(
        self,
        agent_name: str,
        version: str = "v1",
        tenant_id: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Get prompt content for an agent.

        Args:
            agent_name: Name of the agent (e.g., "IntentAgent")
            version: Prompt version (default: "v1")
            tenant_id: Optional tenant identifier for tenant-specific prompts
                       Looks for prompts/{agent_name}/{tenant_id}/{version}.txt first
            use_cache: Whether to use cached prompts

        Returns:
            Prompt content string

        Raises:
            FileNotFoundError: If prompt file not found
        """
        cache_key = (agent_name, version, tenant_id) if tenant_id else (agent_name, version)

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Try tenant-specific prompt first if tenant_id is provided
        if tenant_id:
            tenant_prompt_path = self.prompts_dir / agent_name / tenant_id / f"{version}.txt"
            if tenant_prompt_path.exists():
                content = self._read_prompt_file(tenant_prompt_path)
                if use_cache:
                    self._cache[cache_key] = content
                return content

        # Try default prompt
        default_prompt_path = self.prompts_dir / agent_name / f"{version}.txt"
        if default_prompt_path.exists():
            content = self._read_prompt_file(default_prompt_path)
            if use_cache:
                self._cache[cache_key] = content
            return content

        # Prompt not found
        raise FileNotFoundError(
            f"Prompt not found for agent '{agent_name}' version '{version}'. "
            f"Looked in: {default_prompt_path}"
            + (f" and {tenant_prompt_path}" if tenant_id else "")
        )

    def _read_prompt_file(self, file_path: Path) -> str:
        """
        Read prompt content from file.

        Args:
            file_path: Path to prompt file

        Returns:
            Prompt content string

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        except IOError as e:
            raise IOError(f"Error reading prompt file {file_path}: {str(e)}")

    def list_versions(self, agent_name: str) -> list[str]:
        """
        List all available versions for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of version names (e.g., ["v1", "v2"])
        """
        agent_dir = self.prompts_dir / agent_name
        if not agent_dir.exists():
            return []

        versions = []
        for file_path in agent_dir.glob("*.txt"):
            # Extract version from filename (remove .txt extension)
            version = file_path.stem
            versions.append(version)

        return sorted(versions)

    def list_agents(self) -> list[str]:
        """
        List all agents with available prompts.

        Returns:
            List of agent names
        """
        if not self.prompts_dir.exists():
            return []

        agents = []
        for item in self.prompts_dir.iterdir():
            if item.is_dir():
                agents.append(item.name)

        return sorted(agents)

    def clear_cache(self, agent_name: Optional[str] = None) -> None:
        """
        Clear prompt cache.

        Args:
            agent_name: Specific agent to clear, or None to clear all
        """
        if agent_name:
            keys_to_remove = [
                key for key in self._cache
                if key[0] == agent_name
            ]
            for key in keys_to_remove:
                self._cache.pop(key)
        else:
            self._cache.clear()

    def render_prompt(
        self,
        agent_name: str,
        variables: dict,
        version: str = "v1",
        tenant_id: Optional[str] = None
    ) -> str:
        """
        Load and render prompt with variable substitution.

        Supports {{variable}} placeholder syntax.

        Args:
            agent_name: Name of the agent
            variables: Dictionary of variable substitutions
            version: Prompt version
            tenant_id: Optional tenant identifier

        Returns:
            Rendered prompt content
        """
        content = self.get_prompt(agent_name, version, tenant_id)

        # Replace {{variable}} placeholders
        for key, value in variables.items():
            placeholder = "{{" + key + "}}"
            content = content.replace(placeholder, str(value))

        return content


# Global instance
_prompt_loader: Optional[FilePromptLoader] = None


def get_prompt_loader(prompts_dir: str = "prompts") -> FilePromptLoader:
    """
    Get the global FilePromptLoader instance.

    Args:
        prompts_dir: Directory containing prompt files

    Returns:
        FilePromptLoader instance
    """
    global _prompt_loader

    if _prompt_loader is None:
        _prompt_loader = FilePromptLoader(prompts_dir)

    return _prompt_loader
