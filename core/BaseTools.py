from abc import ABC, abstractmethod

class BaseTool(ABC):
    """
    All tools must inherit from this.
    Tools are deterministic and stateless.
    """
    name: str

    @abstractmethod
    def execute(self, payload: dict) -> dict:
        """
        Executes tool with input payload.
        Must return structured dict result.
        Must NOT mutate global state.
        """
        pass