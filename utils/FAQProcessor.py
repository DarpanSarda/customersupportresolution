"""
FAQProcessor - Extract Q&A pairs and embed only the question.
"""

import re
from typing import List, Dict


class FAQProcessor:
    """Extract Q&A pairs from FAQ files."""

    async def process_file(self, file_path: str, original_filename: str = None) -> List[Dict]:
        """Extract Q&A pairs from FAQ file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract Q&A pairs using regex
        qa_pairs = self._extract_qa_pairs(content)

        filename = original_filename if original_filename else file_path

        # Format as documents
        return [
            {
                "id": f"{filename}_{i}",
                "text": question,  # Embed only the question
                "metadata": {
                    "answer": answer,  # Store answer in metadata
                    "source": filename,
                    "filename": filename
                }
            }
            for i, (question, answer) in enumerate(qa_pairs)
        ]

    def _extract_qa_pairs(self, content: str) -> List[tuple]:
        """Extract Q&A pairs from content."""
        qa_pairs = []

        # Pattern 1: ### Q: ... ? A: ...
        pattern1 = r'###?\s*[Qq]:\s*(.+?)\s*[Aa]:\s*(.+?)(?=###?\s*[Qq]:|$)'
        matches = re.findall(pattern1, content, re.DOTALL)
        qa_pairs.extend([(q.strip(), a.strip()) for q, a in matches])

        # Pattern 2: ## Q: ... \n A: ...
        pattern2 = r'##\s*[Qq]:\s*(.+?)\n\s*[Aa]:\s*(.+?)(?=##\s*[Qq]:|$)'
        matches = re.findall(pattern2, content, re.DOTALL)
        qa_pairs.extend([(q.strip(), a.strip()) for q, a in matches])

        # Pattern 3: Q: ... \nA: ...
        pattern3 = r'(?m)^[Qq]:\s*(.+?)$\s*^[Aa]:\s*(.+?)$'
        matches = re.findall(pattern3, content, re.MULTILINE)
        qa_pairs.extend([(q.strip(), a.strip()) for q, a in matches])

        return qa_pairs
