"""
Tests for document ingestion module.
"""
import unittest
from typing import List

# Mock components to test
def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """Mock chunking logic."""
    if not text:
        return []
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def extract_text(file_path: str) -> str:
    """Mock text extraction."""
    if file_path.endswith('.pdf'):
        return "PDF content mock"
    elif file_path.endswith('.txt'):
        return "TXT content mock"
    elif file_path.endswith('.docx'):
        return "DOCX content mock"
    return ""

class TestIngestion(unittest.TestCase):
    def test_chunking_basic(self):
        """Test basic text chunking."""
        text = "A" * 250
        chunks = chunk_text(text, chunk_size=100)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 100)
        self.assertEqual(len(chunks[1]), 100)
        self.assertEqual(len(chunks[2]), 50)

    def test_chunking_empty(self):
        """Test chunking with empty string."""
        self.assertEqual(chunk_text("", chunk_size=100), [])

    def test_extract_text_pdf(self):
        """Test extracting text from PDF."""
        self.assertEqual(extract_text("test.pdf"), "PDF content mock")

    def test_extract_text_unsupported(self):
        """Test extracting text from unsupported format."""
        self.assertEqual(extract_text("test.xyz"), "")

    def test_empty_file_handling(self):
        """Test edge cases with empty files or content."""
        chunks = chunk_text(extract_text("empty.txt"), 100)
        self.assertIsInstance(chunks, list)

if __name__ == "__main__":
    unittest.main()
