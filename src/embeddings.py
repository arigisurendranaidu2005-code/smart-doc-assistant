"""Embeddings management using HuggingFace sentence-transformers."""
import logging
from typing import List
import torch

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages document and query embeddings."""

    def __init__(self):
        """Initialize the embedding model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.embedding_model
        
        logger.info(f"Initializing embeddings on {self.device} with model {self.model_name}")
        
        try:
            self.embeddings = self._load_model(self.model_name)
        except Exception as e:
            logger.warning(f"Failed to load primary model {self.model_name}: {e}. Falling back to {settings.fallback_embedding_model}")
            self.model_name = settings.fallback_embedding_model
            self.embeddings = self._load_model(self.model_name)

    def _load_model(self, model_name: str) -> HuggingFaceEmbeddings:
        """Load a HuggingFace embeddings model."""
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embeddings(self) -> Embeddings:
        """Return the Langchain Embeddings object."""
        return self.embeddings
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""
        return self.embeddings.embed_documents(texts)
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embeddings.embed_query(text)
