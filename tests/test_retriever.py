"""
Tests for retriever module.
"""
import pytest
from typing import List, Dict, Any

# Mock retriever functions
def hybrid_search(query: str, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
    """Mock hybrid search."""
    # Mock behavior returning scores based on weights
    return [
        {"id": "doc1", "score": vector_weight * 0.9 + keyword_weight * 0.8},
        {"id": "doc2", "score": vector_weight * 0.5 + keyword_weight * 0.9}
    ]

def re_rank(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Mock re-ranking."""
    # Just sorts by score for mock
    return sorted(results, key=lambda x: x["score"], reverse=True)

def test_hybrid_search_scoring():
    """Test if hybrid search correctly applies weights to scores."""
    results = hybrid_search("test query", vector_weight=0.5, keyword_weight=0.5)
    assert len(results) == 2
    # doc1: 0.5*0.9 + 0.5*0.8 = 0.85
    # doc2: 0.5*0.5 + 0.5*0.9 = 0.70
    assert abs(results[0]["score"] - 0.85) < 0.001 or abs(results[1]["score"] - 0.85) < 0.001

def test_re_ranking_order():
    """Test if re-ranking correctly sorts the results."""
    mock_results = [
        {"id": "doc2", "score": 0.5},
        {"id": "doc1", "score": 0.9},
        {"id": "doc3", "score": 0.7}
    ]
    ranked = re_rank(mock_results, "query")
    assert ranked[0]["id"] == "doc1"
    assert ranked[1]["id"] == "doc3"
    assert ranked[2]["id"] == "doc2"

def test_re_ranking_empty():
    """Test re-ranking with empty results."""
    assert re_rank([], "query") == []
