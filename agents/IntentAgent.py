from core.BaseAgent import BaseAgent, AgentExecutionContext
from llms.OutputParser import OutputParser
from models.sections import IntentModel
from models.patch import Patch


class IntentAgent(BaseAgent):
    agent_name = "IntentAgent"
    allowed_section = "understanding"

    def __init__(self, config: dict, prompt: str):
        """Initialize IntentAgent.

        Args:
            config: Agent config containing dependencies
            prompt: Prompt version
        """
        super().__init__(config, prompt)
        self.config_loader = config.get("config_loader")
        self.prompt_loader = config.get("prompt_loader")
        self.llm_client = config.get("llm_client")
        self.parser = OutputParser(IntentModel)

    def _run(self, state: dict, context: AgentExecutionContext) -> Patch:
        """Run intent classification.

        Args:
            state: Current state
            context: Execution context

        Returns:
            Patch with intent classification results
        """
        # -------------------------------------------------
        # 1️⃣ Extract user message from state
        # -------------------------------------------------
        user_message = state.get("conversation", {}).get("latest_message")

        if not user_message:
            raise ValueError("No user message found in state")

        # -------------------------------------------------
        # 2️⃣ Get intent labels
        # -------------------------------------------------
        intent_labels = self.config_loader.get_intent_labels()

        # -------------------------------------------------
        # 3️⃣ Load and render prompt
        # -------------------------------------------------
        template = self.prompt_loader.load(
            agent_name="intent",
            version=self._prompt
        )

        rendered_prompt = self.prompt_loader.render(
            template=template,
            variables={
                "intent_labels": "\n".join(intent_labels),
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

        intent_name = parsed_output.name
        confidence = parsed_output.confidence

        # -------------------------------------------------
        # 6️⃣ Build Patch
        # -------------------------------------------------
        return Patch(
            agent_name=self.agent_name,
            target_section=self.allowed_section,
            confidence=confidence,
            changes={
                "intent": {
                    "name": intent_name,
                    "confidence": confidence
                },
                "input": {
                    "raw_text": user_message
                }
            }
        )
