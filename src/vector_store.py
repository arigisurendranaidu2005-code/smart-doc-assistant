"""Vector store management using ChromaDB."""
import logging
import os
from typing import List, Dict, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from src.embeddings import EmbeddingManager
from src.config import settings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the ChromaDB vector store."""

    def __init__(self, collection_name: str = "smart_doc_assistant"):
        """Initialize the vector store."""
        self.persist_directory = settings.chroma_persist_dir
        self.collection_name = collection_name
        self.embedding_manager = EmbeddingManager()
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        logger.info(f"Initializing Chroma DB at {self.persist_directory}")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_manager.get_embeddings(),
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            logger.warning("No documents to add.")
            return []
            
        logger.info(f"Adding {len(documents)} documents to vector store.")
        try:
            ids = self.vector_store.add_documents(documents)
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform standard similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 20) -> List[Document]:
        """Perform Maximal Marginal Relevance search."""
        return self.vector_store.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

    def delete_collection(self):
        """Delete the collection and its contents."""
        logger.warning(f"Deleting collection {self.collection_name}")
        self.vector_store.delete_collection()
        # Reinitialize an empty store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_manager.get_embeddings(),
            persist_directory=self.persist_directory
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.vector_store._collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}
