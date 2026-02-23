from abc import ABC, abstractmethod
from typing import Dict, Any, List
from models.llm import LLMResponse

class LLMManager(ABC):

    @abstractmethod
    def generate(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        pass
        