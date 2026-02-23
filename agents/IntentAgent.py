from core.BaseAgent import BaseAgent, AgentExecutionContext
from llms.OutputParser import OutputParser
from models.intent import IntentAgentOutput
from models.patch import Patch

class IntentAgent(BaseAgent):
    def __init__(
        self,
        config_loader,
        prompt_loader,
        llm_client
    ):
        self.config_loader = config_loader
        self.prompt_loader = prompt_loader
        self.llm_client = llm_client
        self.parser = OutputParser(IntentAgentOutput)
    
    def execute(self, state: dict, context: AgentExecutionContext) -> Patch:

        # -------------------------------------------------
        # 1️⃣ Extract user message from state
        # -------------------------------------------------

        user_message = state.get("conversation", {}).get("latest_message")

        if not user_message:
            raise ValueError("No user message found in state")

        # -------------------------------------------------
        # 2️⃣ Load config
        # -------------------------------------------------

        agent_config = self.config_loader.get_agent_config("IntentAgent")
        prompt_version = agent_config.get("prompt_version", "v1")

        intent_labels = self.config_loader.get_intent_labels()

        # -------------------------------------------------
        # 3️⃣ Load and render prompt
        # -------------------------------------------------

        template = self.prompt_loader.load(
            agent_name="intent",
            version=prompt_version
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

        intent_name = parsed_output.intent
        confidence = parsed_output.confidence

        # -------------------------------------------------
        # 6️⃣ Build Patch
        # -------------------------------------------------

        patch = Patch(
            agent_name="IntentAgent",
            target_section="understanding",
            confidence=confidence,
            changes={
                "intent": {
                    "name": intent_name,
                    "confidence": confidence
                },
                "input": {
                    "raw_text": user_message
                }
            },
            metadata={
                "execution_time_ms": 0,  # Can measure later
                "config_version": context.config_version,
                "prompt_version": prompt_version,
                "trace_id": context.trace_id,
                "request_id": context.request_id
            }
        )

        return patch
