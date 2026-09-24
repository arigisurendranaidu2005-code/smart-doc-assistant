"""
Tests for retriever module.
"""
import unittest
from typing import List, Dict, Any

# Mock retriever functions
def hybrid_search(query: str, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> List[Dict[str, Any]]:
    """Mock hybrid search."""
    return [
        {"id": "doc1", "score": vector_weight * 0.9 + keyword_weight * 0.8},
        {"id": "doc2", "score": vector_weight * 0.5 + keyword_weight * 0.9}
    ]

def re_rank(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Mock re-ranking."""
    return sorted(results, key=lambda x: x["score"], reverse=True)

class TestRetriever(unittest.TestCase):
    def test_hybrid_search_scoring(self):
        """Test if hybrid search correctly applies weights to scores."""
        results = hybrid_search("test query", vector_weight=0.5, keyword_weight=0.5)
        self.assertEqual(len(results), 2)
        score1 = round(results[0]["score"], 2)
        score2 = round(results[1]["score"], 2)
        self.assertTrue(score1 == 0.85 or score2 == 0.85)

    def test_re_ranking_order(self):
        """Test if re-ranking correctly sorts the results."""
        mock_results = [
            {"id": "doc2", "score": 0.5},
            {"id": "doc1", "score": 0.9},
            {"id": "doc3", "score": 0.7}
        ]
        ranked = re_rank(mock_results, "query")
        self.assertEqual(ranked[0]["id"], "doc1")
        self.assertEqual(ranked[1]["id"], "doc3")
        self.assertEqual(ranked[2]["id"], "doc2")

    def test_re_ranking_empty(self):
        """Test re-ranking with empty results."""
        self.assertEqual(re_rank([], "query"), [])

if __name__ == "__main__":
    unittest.main()
