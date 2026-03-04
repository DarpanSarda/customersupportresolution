"""Document parser for multiple file formats with OCR support.

Supports:
- PDF (unscanned: PyMuPDF, scanned: Doctr OCR)
- DOCX (Word documents)
- TXT (text files)
- HTML (web pages)
- MD (markdown files)

Auto-detects scanned PDFs and uses OCR automatically.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parse documents from multiple file formats."""

    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".txt": "text",
        ".html": "html",
        ".htm": "html",
        ".md": "markdown",
    }

    def __init__(self):
        self._pdf_parser = None
        self._docx_parser = None
        self._doctr_model = None

    def parse_file(self, file_path: str) -> str:
        """Parse a file and extract text content.

        Args:
            file_path: Path to the file

        Returns:
            str: Extracted text content

        Raises:
            ValueError: If file format not supported
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {ext}")

        format_type = self.SUPPORTED_FORMATS[ext]

        if format_type == "pdf":
            return self._parse_pdf(file_path)
        elif format_type == "docx":
            return self._parse_docx(file_path)
        elif format_type == "text":
            return self._parse_text(file_path)
        elif format_type == "html":
            return self._parse_html(file_path)
        elif format_type == "markdown":
            return self._parse_markdown(file_path)

    def _parse_pdf(self, file_path: str) -> str:
        """Parse PDF using PyMuPDF with Doctr OCR fallback for scanned PDFs.

        Strategy:
        1. Try direct text extraction (unscanned PDF)
        2. If no text found, use Doctr OCR (scanned PDF)
        """
        if self._pdf_parser is None:
            try:
                import fitz
                self._pdf_parser = fitz
            except ImportError:
                raise ImportError(
                    "PyMuPDF (fitz) is required for PDF parsing. "
                    "Install with: pip install pymupdf"
                )

        doc = self._pdf_parser.open(file_path)
        text_parts = []

        # First try: Direct text extraction
        for page in doc:
            page_text = page.get_text()
            text_parts.append(page_text)

        full_text = "\n\n".join(text_parts)

        # Check if PDF is scanned (no extractable text)
        if len(full_text.strip()) < 50:
            logger.info(f"Detected scanned PDF: {file_path}, using Doctr OCR...")
            doc.close()
            return self._parse_pdf_with_ocr(file_path)

        doc.close()
        return full_text

    def _parse_pdf_with_ocr(self, file_path: str) -> str:
        """Parse scanned PDF using Doctr.

        Doctr is a deep learning OCR toolkit by Mindee:
        - 100% free, no API keys required
        - Runs locally (no cloud service)
        - High accuracy on documents and tables
        - Pre-trained models

        Install: pip install "doctr[torch]"
        """
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor

        # Initialize Doctr model (lazy load, only when needed)
        if self._doctr_model is None:
            self._doctr_model = ocr_predictor(
                pretrained=True,
                assume_straight_pages=False,  # Handle rotated text
                detect_orientation=True
            )

        # Convert PDF to images using PyMuPDF
        if self._pdf_parser is None:
            import fitz
            self._pdf_parser = fitz

        doc = self._pdf_parser.open(file_path)
        ocr_text_parts = []

        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            # OCR with Doctr
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_bytes))

            # Process single page
            result = self._doctr_model([img])

            # Extract text from results
            page_lines = []
            for page_result in result.pages:
                for block in page_result.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        page_lines.append(line_text)

            page_text = "\n".join(page_lines)
            ocr_text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")

        doc.close()
        return "\n\n".join(ocr_text_parts)

    def _parse_docx(self, file_path: str) -> str:
        """Parse DOCX using python-docx."""
        if self._docx_parser is None:
            try:
                from docx import Document
                self._docx_parser = Document
            except ImportError:
                raise ImportError(
                    "python-docx is required for DOCX parsing. "
                    "Install with: pip install python-docx"
                )

        doc = self._docx_parser(file_path)
        text_parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        # Also extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    text_parts.append(row_text)

        return "\n\n".join(text_parts)

    def _parse_text(self, file_path: str) -> str:
        """Parse plain text file with encoding detection."""
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw = f.read()
                result = chardet.detect(raw)
                encoding = result.get("encoding", "utf-8")

            return raw.decode(encoding, errors="replace")
        except ImportError:
            # Fallback to utf-8
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

    def _parse_html(self, file_path: str) -> str:
        """Parse HTML using BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install with: pip install beautifulsoup4"
            )

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        return soup.get_text(separator="\n\n", strip=True)

    def _parse_markdown(self, file_path: str) -> str:
        """Parse markdown file."""
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


class BatchDocumentParser:
    """Parse multiple documents with progress tracking."""

    def __init__(self):
        self.parser = DocumentParser()

    def parse_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> Dict[str, str]:
        """Parse all supported files in a directory.

        Args:
            directory: Directory path
            extensions: File extensions to include (default: all supported)
            recursive: Whether to search recursively

        Returns:
            Dict mapping file paths to extracted text
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        extensions = extensions or list(DocumentParser.SUPPORTED_FORMATS.keys())
        extensions = set(ext.lower() for ext in extensions)

        results = {}

        if recursive:
            files = dir_path.rglob("*")
        else:
            files = dir_path.glob("*")

        for file_path in files:
            if file_path.suffix.lower() in extensions:
                try:
                    text = self.parser.parse_file(str(file_path))
                    results[str(file_path)] = text
                    logger.info(f"Parsed: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to parse {file_path}: {e}")

        return results


# Convenience function for quick parsing
def parse_document(file_path: str) -> str:
    """Quick helper to parse a single document.

    Args:
        file_path: Path to the document

    Returns:
        str: Extracted text content
    """
    parser = DocumentParser()
    return parser.parse_file(file_path)
