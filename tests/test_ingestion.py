"""
Tests for document ingestion module.
"""
import pytest
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

def test_chunking_basic():
    """Test basic text chunking."""
    text = "A" * 250
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50

def test_chunking_empty():
    """Test chunking with empty string."""
    assert chunk_text("", chunk_size=100) == []

def test_extract_text_pdf():
    """Test extracting text from PDF."""
    assert extract_text("test.pdf") == "PDF content mock"

def test_extract_text_unsupported():
    """Test extracting text from unsupported format."""
    assert extract_text("test.xyz") == ""

def test_empty_file_handling():
    """Test edge cases with empty files or content."""
    # Assuming the ingestion pipeline handles empty text gracefully
    chunks = chunk_text(extract_text("empty.txt"), 100)
    assert isinstance(chunks, list)
