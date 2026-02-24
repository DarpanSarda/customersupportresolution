from core.BaseTools import BaseTool

class FAQLookupTool(BaseTool):

    name = "faq_lookup"

    def __init__(self):
        # In-memory knowledge base (v1)
        self._faq_db = {
            "password reset": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
            "raise a query": "To raise a query, please describe your issue in detail and our support team will assist you shortly.",
            "refund policy": "Our refund policy allows refunds within 7 days of purchase subject to terms and conditions."
        }

    def execute(self, payload: dict) -> dict:

        user_input = payload.get("user_input", "").lower()

        # Simple keyword match (deterministic)
        for keyword, answer in self._faq_db.items():
            if keyword in user_input:
                return {
                    "found": True,
                    "answer": answer
                }

        return {
            "found": False,
            "answer": None
        }