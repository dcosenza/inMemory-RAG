"""Session management utilities for Streamlit app."""

from typing import Any, Optional, List, Dict
import logging

import streamlit as st

from core.rag_pipeline import RAGPipeline
from core.exceptions import RAGChatbotError
from config.settings import get_model_by_display_name, get_available_models, get_default_model

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages Streamlit session state for RAG application."""
    
    @staticmethod
    def initialize_session() -> None:
        """Initialize session state variables."""
        if "initialized" not in st.session_state:
            # Get default model from configuration
            default_model_id, default_display_name = get_default_model()
            
            st.session_state.initialized = True
            st.session_state.rag_pipeline = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.session_state.processing_stats = {}
            st.session_state.current_model_id = default_model_id  # Dynamic default
            st.session_state.current_model_display_name = default_display_name  # Dynamic default
            st.session_state.error_message = None
            st.session_state.success_message = None
            
            logger.info(f"Session initialized with default model: {default_display_name} ({default_model_id})")
    
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
    def set_current_model(display_name: str) -> None:
        """Set current model by display name."""
        try:
            # Get model ID from display name
            model_id = get_model_by_display_name(display_name)
            if not model_id:
                raise ValueError(f"Model not found: {display_name}")
            
            # Update session state
            st.session_state.current_model_display_name = display_name
            st.session_state.current_model_id = model_id
            
            # Update RAG pipeline if available
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.set_model(model_id)
            
            logger.info(f"Current model set to: {display_name} ({model_id})")
            
        except Exception as e:
            logger.error(f"Failed to set model: {e}")
            raise
    
    @staticmethod
    def set_current_model_by_id(model_id: str) -> None:
        """Set current model by model ID."""
        try:
            # Get display name from model ID
            available_models = get_available_models()
            display_name = None
            for name, id_val in available_models.items():
                if id_val == model_id:
                    display_name = name
                    break
            
            if not display_name:
                raise ValueError(f"Model ID not found: {model_id}")
            
            # Update session state
            st.session_state.current_model_id = model_id
            st.session_state.current_model_display_name = display_name
            
            # Update RAG pipeline if available
            if st.session_state.rag_pipeline:
                st.session_state.rag_pipeline.set_model(model_id)
            
            logger.info(f"Current model set to: {display_name} ({model_id})")
            
        except Exception as e:
            logger.error(f"Failed to set model: {e}")
            raise
    
    @staticmethod
    def get_current_model_display_name() -> str:
        """Get current model display name."""
        # Get default if not set
        if "current_model_display_name" not in st.session_state:
            _, default_display_name = get_default_model()
            return default_display_name
        return st.session_state.get("current_model_display_name")
    
    @staticmethod
    def get_current_model_id() -> str:
        """Get current model ID."""
        # Get default if not set
        if "current_model_id" not in st.session_state:
            default_model_id, _ = get_default_model()
            return default_model_id
        return st.session_state.get("current_model_id")
    
    @staticmethod
    def sync_model_with_pipeline() -> None:
        """Sync the current model with the RAG pipeline."""
        try:
            if st.session_state.rag_pipeline:
                current_model_id = st.session_state.get("current_model_id")
                if current_model_id:
                    st.session_state.rag_pipeline.set_model(current_model_id)
                    logger.info(f"Synced model with pipeline: {current_model_id}")
        except Exception as e:
            logger.warning(f"Failed to sync model with pipeline: {e}")
    
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
            "current_model_display_name": st.session_state.get("current_model_display_name", "Unknown"),
            "current_model_id": st.session_state.get("current_model_id", "Unknown"),
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