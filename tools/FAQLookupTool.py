"""FAQ lookup tool for knowledge base queries."""

from typing import Dict, Any, Optional
from core.BaseTools import BaseTool
from models.tool import ToolResult


class FAQLookupTool(BaseTool):
    """
    In-memory FAQ knowledge base lookup.

    Simple keyword-based FAQ matching.
    Can be extended to use vector search, RAG, etc.
    """

    name = "faq_lookup"

    def __init__(self):
        # In-memory knowledge base (v1)
        self._faq_db = {
            "password reset": "To reset your password, click on 'Forgot Password' on the login page and follow the instructions.",
            "raise a query": "To raise a query, please describe your issue in detail and our support team will assist you shortly.",
            "refund policy": "Our refund policy allows refunds within 7 days of purchase subject to terms and conditions.",
            "refund": "Our refund policy allows refunds within 7 days of purchase subject to terms and conditions.",
            "return": "To return an item, go to Your Orders > Select the item > Return or replace items.",
            "shipping": "Standard shipping takes 3-5 business days. Express shipping takes 1-2 business days.",
            "order status": "You can check your order status in the 'My Orders' section of your account.",
            "cancel order": "You can cancel your order within 30 minutes of placing it from the My Orders page.",
            "payment": "We accept credit cards, debit cards, net banking, and UPI payments.",
            "contact": "You can reach our support team at support@example.com or call 1-800-SUPPORT."
        }

    def execute(
        self,
        payload: Dict[str, Any],
        tenant_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Look up FAQ based on user input.

        Args:
            payload: Contains user_input (raw text) or entities
            tenant_id: Tenant identifier (for tenant-specific FAQs)
            context: Execution context

        Returns:
            ToolResult with FAQ answer if found
        """
        # Get user input - try multiple fields
        user_input = (
            payload.get("user_input", "") or
            payload.get("entities", {}).get("query", "") or
            ""
        )

        if not user_input:
            return ToolResult.failed(
                error="No user input provided for FAQ lookup",
                error_code="NO_INPUT"
            )

        user_input_lower = user_input.lower()
        user_words = set(user_input_lower.split())

        print(f"[DEBUG FAQLookupTool] user_input='{user_input}', words={user_words}")

        # Word-based matching for better coverage
        for keyword, answer in self._faq_db.items():
            # Try exact phrase match first
            if keyword in user_input_lower:
                return ToolResult.success(
                    data={
                        "found": True,
                        "answer": answer,
                        "matched_keyword": keyword
                    }
                )

            # Try word-based match (all keyword words present in user input)
            keyword_words = set(keyword.split())
            # For multi-word keywords, require at least 70% of words to match
            if len(keyword_words) > 1:
                match_ratio = len(keyword_words & user_words) / len(keyword_words)
                if match_ratio >= 0.7:
                    return ToolResult.success(
                        data={
                            "found": True,
                            "answer": answer,
                            "matched_keyword": keyword,
                            "match_method": "word_based"
                        }
                    )
            # Single word keywords - just check if word exists
            elif keyword_words.issubset(user_words):
                return ToolResult.success(
                    data={
                        "found": True,
                        "answer": answer,
                        "matched_keyword": keyword,
                        "match_method": "word_based"
                    }
                )

        # No match found
        return ToolResult.success(
            data={
                "found": False,
                "answer": None,
                "suggestion": "I couldn't find a specific answer. Would you like me to connect you to a support agent?"
            }
        )
