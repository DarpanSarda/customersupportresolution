"""
DocumentProcessor - Simple file processor for PDF, MD, TXT.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


class DocumentProcessor:
    """Process files into chunks."""

    SUPPORTED = {".pdf", ".txt", ".md"}

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    async def process_file(self, file_path: str, original_filename: str = None) -> list:
        """Process single file into chunks."""
        path = Path(file_path)

        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(f"Unsupported: {path.suffix}. Use PDF, MD, or TXT")

        # Load file
        if path.suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path))

        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)

        # Use original filename if provided, otherwise use temp file name
        filename = original_filename if original_filename else path.name

        # Format chunks
        return [
            {
                "id": f"{Path(filename).stem}_{i}",
                "text": chunk.page_content,
                "metadata": {"source": filename, "filename": filename}
            }
            for i, chunk in enumerate(chunks)
        ]

    async def process_directory(self, directory_path: str, recursive: bool = True) -> list:
        """Process all files in directory."""
        path = Path(directory_path)
        all_chunks = []

        files = list(path.rglob("*")) if recursive else list(path.glob("*"))
        files = [f for f in files if f.is_file() and f.suffix.lower() in self.SUPPORTED]

        for file_path in files:
            chunks = await self.process_file(str(file_path))
            all_chunks.extend(chunks)

        return all_chunks
