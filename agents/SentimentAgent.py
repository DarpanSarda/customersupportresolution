from core.BaseAgent import BaseAgent, AgentExecutionContext
from llms.OutputParser import OutputParser
from models.sections import SentimentModel
from models.patch import Patch


class SentimentAgent(BaseAgent):
    """
    Analyzes customer sentiment from their message.

    Outputs:
    - label: Primary sentiment category (from config)
    - confidence: Model confidence in prediction (0-1)
    - intensity: Emotional strength (0-1)
    - indicators: Key phrases that led to classification

    All sentiment labels and thresholds are configured via config,
    not hardcoded.
    """

    agent_name = "SentimentAgent"
    allowed_section = "understanding"

    def __init__(self, config: dict, prompt: str):
        """
        Initialize SentimentAgent.

        Args:
            config: Agent config containing dependencies
            prompt: Prompt version
        """
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")
        self.parser = OutputParser(SentimentModel)

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """
        Run sentiment classification on user message.

        Args:
            state: Current state
            context: Execution context

        Returns:
            Patch with sentiment classification results
        """
        # -------------------------------------------------
        # 1️⃣ Extract user message from state
        # -------------------------------------------------
        user_message = state.get("conversation", {}).get("latest_message")

        if not user_message:
            raise ValueError("No user message found in state")

        # -------------------------------------------------
        # 2️⃣ Get sentiment labels from config (not hardcoded)
        # -------------------------------------------------
        sentiment_labels = self.config_loader.get_sentiment_labels()

        if not sentiment_labels:
            raise ValueError("No sentiment labels configured")

        # -------------------------------------------------
        # 3️⃣ Load and render prompt
        # -------------------------------------------------
        template = self.prompt_loader.load(
            agent_name="sentiment",
            version=self._prompt
        )

        rendered_prompt = self.prompt_loader.render(
            template=template,
            variables={
                "sentiment_labels": "\n".join(sentiment_labels),
                "user_message": user_message
            }
        )

        messages = [
            {"role": "system", "content": rendered_prompt}
        ]

        # -------------------------------------------------
        # 4️⃣ Call LLM
        # -------------------------------------------------
        llm_response = self.llm_client.generate(messages)

        # -------------------------------------------------
        # 5️⃣ Parse output
        # -------------------------------------------------
        parsed_output = self.parser.parse(llm_response.content)

        sentiment_label = parsed_output.label
        confidence = parsed_output.confidence
        intensity = parsed_output.intensity or 0.5
        indicators = parsed_output.indicators or []

        # -------------------------------------------------
        # 6️⃣ Build Patch (preserve existing understanding fields)
        # -------------------------------------------------
        # Get existing understanding data to preserve
        existing_understanding = state.get("understanding", {})

        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=confidence,
            changes={
                # Preserve existing fields
                "intent": existing_understanding.get("intent"),
                "input": existing_understanding.get("input"),
                # Add new sentiment data
                "sentiment": {
                    "label": sentiment_label,
                    "confidence": confidence,
                    "intensity": intensity,
                    "indicators": indicators
                }
            }
        )
