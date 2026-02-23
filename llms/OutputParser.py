import json
from pydantic import ValidationError

class OutputParser:

    def __init__(self, schema_model):
        self.schema_model = schema_model

    def parse(self, raw_text: str):
        """
        Extract JSON from LLM response and validate against schema.
        """

        try:
            json_data = self._extract_json(raw_text)
        except Exception as e:
            raise ValueError(f"Failed to extract JSON: {str(e)}")

        try:
            validated = self.schema_model(**json_data)
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {str(e)}")

        return validated

    def _extract_json(self, text: str):
        """
        Minimal JSON extraction.
        Assumes LLM returns a JSON object somewhere in text.
        """

        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise ValueError("No JSON object found")

        json_str = text[start:end+1]

        return json.loads(json_str)