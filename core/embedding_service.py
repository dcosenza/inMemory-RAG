"""Embedding service for document and query vectorization."""

from typing import List, Optional
import logging
import os

import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np

# Set environment variables to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config.settings import EMBEDDING_CONFIG
from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Handles text embedding generation using sentence transformers."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize embedding service with configuration."""
        self.config = config or EMBEDDING_CONFIG
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load and cache the embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @st.cache_resource(show_spinner="Loading embedding model...")
    def _load_model(_self) -> SentenceTransformer:
        """Load the sentence transformer model with caching."""
        try:
            model = SentenceTransformer(
                _self.config.model_name,
                device=_self.config.device
            )
            logger.info(f"Loaded embedding model: {_self.config.model_name}")
            return model
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of documents."""
        if not texts:
            return np.array([])
        
        try:
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                raise EmbeddingError("No valid texts to embed")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.batch_size,
                show_progress_bar=len(valid_texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=False  # We'll normalize in vector store for FAISS
            )
            
            logger.info(f"Generated embeddings for {len(valid_texts)} documents")
            return embeddings
        
        except Exception as e:
            raise EmbeddingError(f"Failed to generate document embeddings: {e}")
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        if not query or not query.strip():
            raise EmbeddingError("Query cannot be empty")
        
        try:
            embedding = self.model.encode(
                [query.strip()],
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            
            return embedding[0]  # Return single embedding vector
        
        except Exception as e:
            raise EmbeddingError(f"Failed to generate query embedding: {e}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.warning(f"Could not get embedding dimension: {e}")
            # Default dimension for all-MiniLM-L6-v2
            return 384
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.config.model_name,
            "device": self.config.device,
            "embedding_dimension": self.get_embedding_dimension(),
            "batch_size": self.config.batch_size,
            "model_loaded": self._model is not None
        }