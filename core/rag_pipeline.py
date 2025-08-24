"""RAG Pipeline orchestrating document processing, embedding, and retrieval."""

from typing import List, Tuple, Optional, Iterator, Dict, Any
import logging

from langchain.schema import Document

from core.document_processor import DocumentProcessor
from core.embedding_service import EmbeddingService
from core.vector_store import FAISSVectorStore
from core.llm_service import LLMService
from core.exceptions import RAGChatbotError
from config.settings import MODEL_CONFIG, EMBEDDING_CONFIG, VECTOR_STORE_CONFIG

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self):
        """Initialize RAG pipeline with all components."""
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.vector_store: Optional[FAISSVectorStore] = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> None:
        """Initialize vector store with embedding dimension."""
        try:
            embedding_dimension = self.embedding_service.get_embedding_dimension()
            self.vector_store = FAISSVectorStore(embedding_dimension)
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            raise RAGChatbotError(f"Failed to initialize RAG pipeline: {e}")
    
    def set_model(self, model_name: str) -> None:
        """Set the current model for the LLM service."""
        try:
            self.llm_service.set_model(model_name)
            logger.info(f"Model set to: {model_name}")
        except Exception as e:
            raise RAGChatbotError(f"Failed to set model: {e}")
    
    def set_model_by_display_name(self, display_name: str) -> None:
        """Set model by its display name."""
        try:
            self.llm_service.set_model_by_display_name(display_name)
            logger.info(f"Model set to: {display_name}")
        except Exception as e:
            raise RAGChatbotError(f"Failed to set model: {e}")
    
    def get_current_model(self) -> str:
        """Get the current model ID."""
        return self.llm_service.get_current_model()
    
    def get_current_model_display_name(self) -> str:
        """Get the current model display name."""
        return self.llm_service.get_current_model_display_name()
    
    def get_available_models(self, include_paid: bool = False) -> Dict[str, str]:
        """Get available models for selection."""
        return self.llm_service.get_available_models(include_paid)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        return self.llm_service.get_model_info(model_name)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return self.llm_service.get_current_model_info()
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if a model is available."""
        return self.llm_service.validate_model(model_name)
    
    def process_documents(self, files_data: List[Tuple[bytes, str]]) -> Dict[str, Any]:
        """Process uploaded documents and add to vector store."""
        if not files_data:
            raise RAGChatbotError("No files provided for processing")
        
        all_documents = []
        processing_stats = {"files_processed": 0, "files_failed": 0, "errors": []}
        
        # Process each file
        for file_data, filename in files_data:
            try:
                documents = self.document_processor.process_pdf(file_data, filename)
                all_documents.extend(documents)
                processing_stats["files_processed"] += 1
                logger.info(f"Successfully processed {filename}")
                
            except Exception as e:
                processing_stats["files_failed"] += 1
                error_msg = f"Failed to process {filename}: {str(e)}"
                processing_stats["errors"].append(error_msg)
                logger.error(error_msg)
        
        if not all_documents:
            raise RAGChatbotError("No valid documents were processed")
        
        # Generate embeddings
        try:
            texts = [doc.page_content for doc in all_documents]
            embeddings = self.embedding_service.embed_documents(texts)
            
            # Add to vector store
            self.vector_store.add_documents(all_documents, embeddings)
            
            # Get final stats
            doc_stats = self.document_processor.get_processing_stats(all_documents)
            vector_stats = self.vector_store.get_stats()
            
            processing_stats.update({
                "documents_created": len(all_documents),
                "total_characters": doc_stats.get("total_characters", 0),
                "vector_store_size": vector_stats.get("total_documents", 0),
                "memory_usage_mb": vector_stats.get("memory_usage_mb", 0)
            })
            
            logger.info(f"Successfully processed {len(all_documents)} document chunks")
            return processing_stats
            
        except Exception as e:
            raise RAGChatbotError(f"Failed to create embeddings and vector store: {e}")
    
    def query(
        self, 
        question: str, 
        model_name: Optional[str] = None,
        max_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> Tuple[str, List[Document], List[float]]:
        """Query the RAG system and return response with source documents."""
        if not question or not question.strip():
            raise RAGChatbotError("Question cannot be empty")
        
        if self.vector_store.is_empty():
            raise RAGChatbotError("No documents have been processed. Please upload documents first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(question)
            
            # Retrieve relevant documents
            logger.info(f"Searching for documents with k={max_results}")
            relevant_docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query_embedding, 
                k=max_results
            )
            
            logger.info(f"Retrieved {len(relevant_docs_with_scores)} documents with scores")
            
            if not relevant_docs_with_scores:
                logger.warning("No relevant documents found - returning empty response")
                return (
                    "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    [],
                    []
                )
            
            # Separate documents and scores
            relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
            scores = [score for _, score in relevant_docs_with_scores]
            
            # Generate response
            response = self.llm_service.generate_response(
                question, 
                relevant_docs, 
                model_name=model_name
            )
            
            logger.info(f"Generated response for query: {question[:50]}...")
            return response, relevant_docs, scores
            
        except Exception as e:
            if isinstance(e, RAGChatbotError):
                raise
            raise RAGChatbotError(f"Failed to process query: {e}")
    
    def query_streaming(
        self,
        question: str,
        model_name: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> Tuple[Iterator[str], List[Document], List[float]]:
        """Query with streaming response."""
        if not question or not question.strip():
            raise RAGChatbotError("Question cannot be empty")
        
        if self.vector_store.is_empty():
            raise RAGChatbotError("No documents have been processed. Please upload documents first.")
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(question)
            
            # Retrieve relevant documents
            logger.info(f"Searching for documents with k={max_results}")
            relevant_docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                query_embedding,
                k=max_results
            )
            
            logger.info(f"Retrieved {len(relevant_docs_with_scores)} documents with scores")
            
            if not relevant_docs_with_scores:
                logger.warning("No relevant documents found - returning empty response")
                def empty_response():
                    yield "I couldn't find any relevant information in the uploaded documents to answer your question."
                
                return empty_response(), [], []
            
            # Separate documents and scores
            relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
            scores = [score for _, score in relevant_docs_with_scores]
            
            # Generate streaming response
            response_stream = self.llm_service.generate_streaming_response(
                question,
                relevant_docs,
                model_name=model_name
            )
            
            logger.info(f"Started streaming response for query: {question[:50]}...")
            return response_stream, relevant_docs, scores
            
        except Exception as e:
            if isinstance(e, RAGChatbotError):
                raise
            raise RAGChatbotError(f"Failed to process streaming query: {e}")
    
    def clear_documents(self) -> None:
        """Clear all documents from the pipeline."""
        try:
            if self.vector_store:
                self.vector_store.delete_all()
            logger.info("Cleared all documents from RAG pipeline")
        except Exception as e:
            raise RAGChatbotError(f"Failed to clear documents: {e}")
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all pipeline components."""
        validation = {}
        
        try:
            # Validate document processor
            validation["document_processor"] = self.document_processor is not None
            
            # Validate embedding service
            validation["embedding_service"] = self.embedding_service is not None
            
            # Validate LLM service
            validation["llm_service"] = self.llm_service is not None
            
            # Validate vector store
            validation["vector_store"] = self.vector_store is not None
            
            # Overall validation - exclude the "overall" key itself
            component_validation = [
                validation["document_processor"],
                validation["embedding_service"],
                validation["llm_service"],
                validation["vector_store"]
            ]
            validation["overall"] = all(component_validation)
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            validation["overall"] = False
        
        return validation
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "pipeline_initialized": True,
            "current_model": self.get_current_model(),
            "current_model_display_name": self.get_current_model_display_name(),
            "available_models_count": len(self.get_available_models()),
            "vector_store_initialized": self.vector_store is not None
        }
        
        # Add vector store stats if available
        if self.vector_store:
            try:
                vector_stats = self.vector_store.get_stats()
                stats.update(vector_stats)
            except Exception as e:
                logger.warning(f"Could not get vector store stats: {e}")
                stats["vector_store_stats_error"] = str(e)
        
        # Add model info
        try:
            model_info = self.get_current_model_info()
            stats["model_info"] = model_info
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            stats["model_info_error"] = str(e)
        
        return stats