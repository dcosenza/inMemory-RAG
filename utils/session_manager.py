"""Session management utilities for Streamlit app."""

from typing import Any, Optional, List, Dict
import logging

import streamlit as st

from core.rag_pipeline import RAGPipeline
from core.exceptions import RAGChatbotError

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages Streamlit session state for RAG application."""
    
    @staticmethod
    def initialize_session() -> None:
        """Initialize session state variables."""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.session_state.processing_stats = {}
            st.session_state.current_model = "Mistral 7B Instruct"
            st.session_state.error_message = None
            st.session_state.success_message = None
            
            logger.info("Session initialized")
    
    @staticmethod
    def get_rag_pipeline() -> RAGPipeline:
        """Get or create RAG pipeline instance."""
        if st.session_state.rag_pipeline is None:
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                logger.info("Created new RAG pipeline instance")
            except Exception as e:
                error_msg = f"Failed to initialize RAG pipeline: {e}"
                logger.error(error_msg)
                st.error(error_msg)
                raise RAGChatbotError(error_msg)
        
        return st.session_state.rag_pipeline
    
    @staticmethod
    def add_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None) -> None:
        """Add message to chat history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": st.session_state.get("current_timestamp"),
            "sources": sources or []
        }
        
        st.session_state.chat_history.append(message)
        
        # Limit chat history to prevent memory issues
        max_history = 50
        if len(st.session_state.chat_history) > max_history:
            st.session_state.chat_history = st.session_state.chat_history[-max_history:]
    
    @staticmethod
    def get_chat_history() -> List[Dict[str, Any]]:
        """Get chat history."""
        return st.session_state.get("chat_history", [])
    
    @staticmethod
    def clear_chat_history() -> None:
        """Clear chat history."""
        st.session_state.chat_history = []
        logger.info("Chat history cleared")
    
    @staticmethod
    def set_documents_processed(status: bool, stats: Optional[Dict] = None) -> None:
        """Set document processing status."""
        st.session_state.documents_processed = status
        if stats:
            st.session_state.processing_stats = stats
        
        logger.info(f"Documents processed status: {status}")
    
    @staticmethod
    def is_documents_processed() -> bool:
        """Check if documents are processed."""
        return st.session_state.get("documents_processed", False)
    
    @staticmethod
    def get_processing_stats() -> Dict[str, Any]:
        """Get document processing statistics."""
        return st.session_state.get("processing_stats", {})
    
    @staticmethod
    def set_current_model(model_name: str) -> None:
        """Set current model."""
        st.session_state.current_model = model_name
        logger.info(f"Current model set to: {model_name}")
    
    @staticmethod
    def get_current_model() -> str:
        """Get current model."""
        return st.session_state.get("current_model", "Mistral 7B Instruct")
    
    @staticmethod
    def set_error_message(message: str) -> None:
        """Set error message."""
        st.session_state.error_message = message
        logger.error(f"Error message set: {message}")
    
    @staticmethod
    def get_error_message() -> Optional[str]:
        """Get and clear error message."""
        message = st.session_state.get("error_message")
        if message:
            st.session_state.error_message = None
        return message
    
    @staticmethod
    def set_success_message(message: str) -> None:
        """Set success message."""
        st.session_state.success_message = message
        logger.info(f"Success message set: {message}")
    
    @staticmethod
    def get_success_message() -> Optional[str]:
        """Get and clear success message."""
        message = st.session_state.get("success_message")
        if message:
            st.session_state.success_message = None
        return message
    
    @staticmethod
    def reset_session() -> None:
        """Reset session state (except for initialization flag)."""
        try:
            # Clear RAG pipeline
            if st.session_state.get("rag_pipeline"):
                st.session_state.rag_pipeline.clear_documents()
            
            # Reset state variables
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.session_state.processing_stats = {}
            st.session_state.error_message = None
            st.session_state.success_message = None
            
            logger.info("Session reset successfully")
            
        except Exception as e:
            error_msg = f"Failed to reset session: {e}"
            logger.error(error_msg)
            st.error(error_msg)
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get comprehensive session information."""
        pipeline = st.session_state.get("rag_pipeline")
        
        info = {
            "session_initialized": st.session_state.get("initialized", False),
            "documents_processed": st.session_state.get("documents_processed", False),
            "chat_history_length": len(st.session_state.get("chat_history", [])),
            "current_model": st.session_state.get("current_model", "Unknown"),
            "processing_stats": st.session_state.get("processing_stats", {}),
            "pipeline_initialized": pipeline is not None
        }
        
        if pipeline:
            try:
                pipeline_stats = pipeline.get_pipeline_stats()
                info["pipeline_stats"] = pipeline_stats
            except Exception as e:
                logger.warning(f"Could not get pipeline stats: {e}")
                info["pipeline_stats"] = {"error": str(e)}
        
        return info