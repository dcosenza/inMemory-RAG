"""Vector store implementation using FAISS for similarity search."""

from typing import List, Tuple, Optional, Dict, Any
import logging

import numpy as np
import faiss
from langchain.schema import Document

from config.settings import VECTOR_STORE_CONFIG
from core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, embedding_dimension: int, config: Optional[dict] = None):
        """Initialize FAISS vector store."""
        self.config = config or VECTOR_STORE_CONFIG
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.documents: List[Document] = []
        self.document_embeddings: Optional[np.ndarray] = None
        
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize FAISS index for similarity search."""
        try:
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(self.embedding_dimension)
            logger.info(f"Initialized FAISS index with dimension {self.embedding_dimension}")
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize FAISS index: {e}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for FAISS."""
        embeddings_array = np.array(embeddings, dtype=np.float32)
        if embeddings_array.ndim == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        # Ensure embeddings are properly normalized for FAISS
        faiss.normalize_L2(embeddings_array)
        return embeddings_array
    
    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        """Add documents and their embeddings to the vector store."""
        if len(documents) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between documents ({len(documents)}) and embeddings ({len(embeddings)})"
            )
        
        if not documents:
            logger.warning("No documents to add to vector store")
            return
        
        try:
            # Normalize and add embeddings to FAISS index
            embeddings_array = self._normalize_embeddings(embeddings)
            self.index.add(embeddings_array)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            if self.document_embeddings is None:
                self.document_embeddings = embeddings_array
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings_array])
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents to vector store: {e}")
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search and return documents with scores."""
        k = k or self.config.max_results
        score_threshold = score_threshold or self.config.similarity_threshold
        
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return []
        
        try:
            # Normalize query embedding
            query_array = self._normalize_embeddings(query_embedding)
            
            # Perform search
            scores, indices = self.index.search(query_array, min(k, self.index.ntotal))
            
            # Filter results by similarity threshold and return documents with scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:  # -1 indicates no match found
                    document = self.documents[idx]
                    results.append((document, float(score)))
            
            logger.info(f"Found {len(results)} relevant documents (threshold: {score_threshold})")
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to perform similarity search: {e}")
    
    def similarity_search_with_relevance_scores(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search and return results sorted by relevance."""
        results = self.similarity_search(query_embedding, k)
        
        # Sort by score in descending order (higher score = more similar)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_relevant_documents(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Get relevant documents without scores."""
        results = self.similarity_search(query_embedding, k, score_threshold)
        return [doc for doc, _ in results]
    
    def delete_all(self) -> None:
        """Clear all documents and reinitialize the index."""
        try:
            self._initialize_index()
            self.documents.clear()
            self.document_embeddings = None
            logger.info("Cleared all documents from vector store")
        except Exception as e:
            raise VectorStoreError(f"Failed to clear vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dimension,
            "index_size": self.index.ntotal if self.index else 0,
            "similarity_threshold": self.config.similarity_threshold,
            "max_results": self.config.max_results,
            "memory_usage_mb": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        if not self.documents or self.document_embeddings is None:
            return 0.0
        
        # Rough estimation: embeddings + documents metadata
        embeddings_size = self.document_embeddings.nbytes
        docs_size = sum(len(doc.page_content.encode('utf-8')) for doc in self.documents)
        
        total_bytes = embeddings_size + docs_size
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def is_empty(self) -> bool:
        """Check if the vector store is empty."""
        return len(self.documents) == 0