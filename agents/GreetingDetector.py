"""
GreetingDetector - Fast greeting detection using MNLI model from env.

Uses CLASSIFICATION_MODEL env variable with 0.85 threshold.
"""

import os
from typing import Optional, Dict, Any
import torch
from transformers import pipeline


class GreetingDetector:
    """Greeting detection using MNLI model from CLASSIFICATION_MODEL env var."""

    def __init__(self):
        """Initialize GreetingDetector."""
        self.model_name = os.getenv("CLASSIFICATION_MODEL")
        self.device = 0 if torch.cuda.is_available() else -1
        self._classifier = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._classifier is None:
            print(f"Loading greeting detection model: {self.model_name}")
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=self.device
            )
            print("Greeting detection model loaded.")

    @property
    def classifier(self):
        """Get classifier, loading if necessary."""
        if self._classifier is None:
            self._load_model()
        return self._classifier

    def is_greeting(self, text: str, threshold: float = 0.85) -> tuple[bool, float]:
        """
        Check if text is a greeting using zero-shot classification.

        Args:
            text: User message to check
            threshold: Confidence threshold (default 0.85)

        Returns:
            (is_greeting, confidence) tuple
        """
        if not text or len(text.strip()) < 2:
            return False, 0.0

        try:
            # Use zero-shot classification with candidate labels
            candidate_labels = ["greeting", "other query"]
            result = self.classifier(text, candidate_labels)

            # Find the greeting label score
            labels = result.get("labels", [])
            scores = result.get("scores", [])

            greeting_idx = labels.index("greeting") if "greeting" in labels else -1
            if greeting_idx == -1:
                return False, 0.0

            greeting_prob = scores[greeting_idx]
            is_greeting = greeting_prob >= threshold
            return is_greeting, greeting_prob

        except Exception as e:
            print(f"Greeting detection error: {e}")
            return False, 0.0

    def get_greeting_response(self) -> str:
        """Get simple greeting response."""
        return "Hello! How can I help you today?"

    def detect_and_respond(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Detect if text is greeting and return response if so.

        Args:
            text: User message

        Returns:
            Dict with greeting response if detected, None otherwise
        """
        is_greeting, confidence = self.is_greeting(text)

        if is_greeting:
            return {
                "is_greeting": True,
                "confidence": confidence,
                "response": self.get_greeting_response(),
                "skip_pipeline": True
            }

        return None


# Singleton instance
_greeting_detector: Optional[GreetingDetector] = None


def get_greeting_detector() -> GreetingDetector:
    """Get singleton GreetingDetector instance."""
    global _greeting_detector
    if _greeting_detector is None:
        _greeting_detector = GreetingDetector()
    return _greeting_detector
