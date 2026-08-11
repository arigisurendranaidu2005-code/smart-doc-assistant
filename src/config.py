"""Configuration management for the application."""
import os
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""
    
    # Document Ingestion
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    fallback_embedding_model: str = "paraphrase-albert-small-v2"
    
    # Vector Store
    chroma_persist_dir: str = "./chroma_db"
    
    # Retriever
    top_k: int = 5
    hybrid_vector_weight: float = 0.5
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM & Chain
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "llama3"  # default for ollama
    openai_api_key: str | None = None
    temperature: float = 0.0
    memory_window_size: int = 5
    
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
