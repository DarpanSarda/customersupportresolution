from pathlib import Path


class PromptLoader:

    def __init__(self, base_path: str | None = None):
        """Initialize PromptLoader with project root relative path.

        Args:
            base_path: Base path to prompts directory. Defaults to project_root/prompts
        """
        if base_path is None:
            # Get the project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            base_path = project_root / "prompts"
        self.base_path = Path(base_path)

    def load(self, agent_name: str, version: str) -> str:
        """
        Load prompt file based on agent name and version.
        """
        agent_folder = self.base_path / agent_name.lower().replace("agent","")
        file_path = agent_folder / f"{version}.txt"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {file_path}"
            )
        return file_path.read_text()

    def render(self, template: str, variables: dict) -> str:
        """
        Simple placeholder replacement.
        """

        prompt = template

        for key, value in variables.items():
            placeholder = f"{{{{{key}}}}}"
            prompt = prompt.replace(placeholder, str(value))

        return prompt