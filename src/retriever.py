"""Hybrid retriever with cross-encoder re-ranking."""
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

from src.vector_store import VectorStoreManager
from src.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining Vector Search, BM25, and Cross-Encoder Re-ranking."""
    
    vector_store: Any
    bm25_corpus: List[Document] = []
    bm25_model: Any = None
    cross_encoder: Any = None
    k: int = settings.top_k
    vector_weight: float = settings.hybrid_vector_weight
    
    def __init__(self, vector_store: VectorStoreManager, all_docs: List[Document] = None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        
        # Initialize BM25
        if all_docs:
            self._initialize_bm25(all_docs)
            
        # Initialize Cross-Encoder
        if CrossEncoder is not None:
            logger.info(f"Loading cross-encoder model: {settings.cross_encoder_model}")
            self.cross_encoder = CrossEncoder(settings.cross_encoder_model)
        else:
            logger.warning("sentence-transformers not installed, cross-encoder ranking disabled.")

    def _initialize_bm25(self, docs: List[Document]):
        """Initialize BM25 index."""
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed, BM25 disabled. Run: pip install rank_bm25")
            return
            
        self.bm25_corpus = docs
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.bm25_corpus]
        self.bm25_model = BM25Okapi(tokenized_corpus)
        logger.info(f"Initialized BM25 with {len(docs)} documents.")

    def update_bm25(self, new_docs: List[Document]):
        """Update BM25 index with new documents."""
        all_docs = self.bm25_corpus + new_docs
        self._initialize_bm25(all_docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get relevant documents using hybrid approach."""
        
        # 1. Vector Search
        # Fetch more candidates for re-ranking
        fetch_k = max(self.k * 3, 10)
        vector_docs = self.vector_store.vector_store.similarity_search_with_relevance_scores(query, k=fetch_k)
        
        # 2. BM25 Search
        bm25_docs = []
        if self.bm25_model is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            
            # Get top fetch_k BM25 docs
            top_n_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]
            
            # Normalize BM25 scores roughly (0 to 1) for merging
            max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            for i in top_n_indices:
                if bm25_scores[i] > 0:
                    bm25_docs.append((self.bm25_corpus[i], bm25_scores[i] / max_score))

        # Merge candidate pools, deduplicate by source+chunk_index or content
        candidates = {}
        
        for doc, score in vector_docs:
            content = doc.page_content
            candidates[content] = {"doc": doc, "vector_score": score, "bm25_score": 0.0}
            
        for doc, score in bm25_docs:
            content = doc.page_content
            if content in candidates:
                candidates[content]["bm25_score"] = score
            else:
                candidates[content] = {"doc": doc, "vector_score": 0.0, "bm25_score": score}

        # Calculate hybrid score if cross encoder is not available
        docs_to_rank = []
        for content, data in candidates.items():
            hybrid_score = (self.vector_weight * data["vector_score"]) + ((1 - self.vector_weight) * data["bm25_score"])
            data["hybrid_score"] = hybrid_score
            docs_to_rank.append(data["doc"])

        # 3. Cross-Encoder Re-ranking
        if self.cross_encoder is not None and docs_to_rank:
            pairs = [[query, doc.page_content] for doc in docs_to_rank]
            try:
                scores = self.cross_encoder.predict(pairs)
                # Combine doc and new score
                ranked_docs = [(doc, score) for doc, score in zip(docs_to_rank, scores)]
                # Sort by cross-encoder score
                ranked_docs.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in ranked_docs[:self.k]]
            except Exception as e:
                logger.error(f"Error during cross-encoder re-ranking: {e}")
                # Fallback to hybrid score
                pass
                
        # Fallback to sorting by hybrid score
        ranked_docs = sorted(list(candidates.values()), key=lambda x: x["hybrid_score"], reverse=True)
        return [item["doc"] for item in ranked_docs[:self.k]]
