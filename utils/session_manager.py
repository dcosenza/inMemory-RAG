"""Session management utilities for Streamlit app."""

from typing import Any, Optional, List, Dict
import logging
import time
from datetime import datetime, timedelta

import streamlit as st

from core.rag_pipeline import RAGPipeline
from core.exceptions import RAGChatbotError
from config.settings import get_model_by_display_name, get_available_models, get_default_model, APP_CONFIG

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
            st.session_state.current_model_id = default_model_id
            st.session_state.current_model_display_name = default_display_name
            st.session_state.error_message = None
            st.session_state.success_message = None
            
            # TTL tracking
            st.session_state.session_created_at = time.time()
            st.session_state.last_activity = time.time()
            
            logger.info(f"Session initialized with default model: {default_display_name} ({default_model_id})")
        else:
            # Update last activity for existing sessions
            SessionManager._update_last_activity()
    
    @staticmethod
    def _update_last_activity() -> None:
        """Update the last activity timestamp."""
        if "last_activity" in st.session_state:
            st.session_state.last_activity = time.time()
    
    @staticmethod
    def _get_session_age_minutes() -> float:
        """Get session age in minutes."""
        if "session_created_at" not in st.session_state:
            return 0.0
        return (time.time() - st.session_state.session_created_at) / 60.0
    
    @staticmethod
    def _get_inactivity_minutes() -> float:
        """Get minutes since last activity."""
        if "last_activity" not in st.session_state:
            return 0.0
        return (time.time() - st.session_state.last_activity) / 60.0
    
    @staticmethod
    def is_session_expired() -> bool:
        """Check if the session has expired based on TTL."""
        if not SessionManager._is_session_initialized():
            return False
        
        ttl_minutes = APP_CONFIG.session_ttl_minutes
        session_age = SessionManager._get_session_age_minutes()
        
        return session_age >= ttl_minutes
    
    @staticmethod
    def is_session_expiring_soon() -> bool:
        """Check if the session is expiring soon (within warning period)."""
        if not SessionManager._is_session_initialized():
            return False
        
        ttl_minutes = APP_CONFIG.session_ttl_minutes
        warning_minutes = APP_CONFIG.session_warning_minutes
        session_age = SessionManager._get_session_age_minutes()
        
        return session_age >= (ttl_minutes - warning_minutes)
    
    @staticmethod
    def get_session_ttl_info() -> Dict[str, Any]:
        """Get session TTL information."""
        if not SessionManager._is_session_initialized():
            return {
                "initialized": False,
                "expired": False,
                "expiring_soon": False,
                "session_age_minutes": 0.0,
                "inactivity_minutes": 0.0,
                "ttl_minutes": APP_CONFIG.session_ttl_minutes,
                "warning_minutes": APP_CONFIG.session_warning_minutes,
                "time_remaining_minutes": 0.0
            }
        
        session_age = SessionManager._get_session_age_minutes()
        inactivity = SessionManager._get_inactivity_minutes()
        ttl_minutes = APP_CONFIG.session_ttl_minutes
        time_remaining = max(0, ttl_minutes - session_age)
        
        return {
            "initialized": True,
            "expired": SessionManager.is_session_expired(),
            "expiring_soon": SessionManager.is_session_expiring_soon(),
            "session_age_minutes": round(session_age, 1),
            "inactivity_minutes": round(inactivity, 1),
            "ttl_minutes": ttl_minutes,
            "warning_minutes": APP_CONFIG.session_warning_minutes,
            "time_remaining_minutes": round(time_remaining, 1)
        }
    
    @staticmethod
    def _is_session_initialized() -> bool:
        """Check if session is initialized."""
        return st.session_state.get("initialized", False)
    
    @staticmethod
    def force_session_expiry() -> None:
        """Force session expiry by clearing session state."""
        try:
            # Clear RAG pipeline
            if st.session_state.get("rag_pipeline"):
                st.session_state.rag_pipeline.clear_documents()
            
            # Clear all session state except initialization flag
            keys_to_clear = [
                "rag_pipeline", "chat_history", "documents_processed", 
                "processing_stats", "current_model_id", "current_model_display_name",
                "error_message", "success_message", "session_created_at", "last_activity"
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            
            logger.info("Session forcefully expired")
            
        except Exception as e:
            logger.error(f"Failed to force session expiry: {e}")
    
    @staticmethod
    def get_rag_pipeline() -> RAGPipeline:
        """Get or create RAG pipeline instance."""
        # Check for session expiry
        if SessionManager.is_session_expired():
            logger.warning("Session expired, forcing new session")
            SessionManager.force_session_expiry()
            st.rerun()
        
        if st.session_state.rag_pipeline is None:
            try:
                st.session_state.rag_pipeline = RAGPipeline()
                SessionManager._update_last_activity()
                logger.info("Created new RAG pipeline instance")
            except Exception as e:
                error_msg = f"Failed to initialize RAG pipeline: {e}"
                logger.error(error_msg)
                st.error(error_msg)
                raise RAGChatbotError(error_msg)
        
        return st.session_state.rag_pipeline
    
    # Chat history methods
    @staticmethod
    def add_chat_message(role: str, content: str, sources: Optional[List[Dict]] = None) -> None:
        """Add message to chat history."""
        SessionManager._update_last_activity()
        
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
        SessionManager._update_last_activity()
        return st.session_state.get("chat_history", [])
    
    @staticmethod
    def clear_chat_history() -> None:
        """Clear chat history."""
        SessionManager._update_last_activity()
        st.session_state.chat_history = []
        logger.info("Chat history cleared")
    
    # Document processing methods
    @staticmethod
    def set_documents_processed(status: bool, stats: Optional[Dict] = None) -> None:
        """Set document processing status."""
        SessionManager._update_last_activity()
        st.session_state.documents_processed = status
        if stats:
            st.session_state.processing_stats = stats
        
        logger.info(f"Documents processed status: {status}")
    
    @staticmethod
    def is_documents_processed() -> bool:
        """Check if documents are processed."""
        SessionManager._update_last_activity()
        return st.session_state.get("documents_processed", False)
    
    @staticmethod
    def get_processing_stats() -> Dict[str, Any]:
        """Get document processing statistics."""
        SessionManager._update_last_activity()
        return st.session_state.get("processing_stats", {})
    
    # Model management methods
    @staticmethod
    def _set_model_internal(display_name: str, model_id: str) -> None:
        """Internal method to set model in session state."""
        SessionManager._update_last_activity()
        
        # Update session state
        st.session_state.current_model_display_name = display_name
        st.session_state.current_model_id = model_id
        
        # Update RAG pipeline if available
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.set_model(model_id)
        
        logger.info(f"Current model set to: {display_name} ({model_id})")
    
    @staticmethod
    def set_current_model(display_name: str) -> None:
        """Set current model by display name."""
        try:
            # Get model ID from display name
            model_id = get_model_by_display_name(display_name)
            if not model_id:
                raise ValueError(f"Model not found: {display_name}")
            
            SessionManager._set_model_internal(display_name, model_id)
            
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
            
            SessionManager._set_model_internal(display_name, model_id)
            
        except Exception as e:
            logger.error(f"Failed to set model: {e}")
            raise
    
    @staticmethod
    def get_current_model_display_name() -> str:
        """Get current model display name."""
        SessionManager._update_last_activity()
        # Get default if not set
        if "current_model_display_name" not in st.session_state:
            _, default_display_name = get_default_model()
            return default_display_name
        return st.session_state.get("current_model_display_name")
    
    @staticmethod
    def get_current_model_id() -> str:
        """Get current model ID."""
        SessionManager._update_last_activity()
        # Get default if not set
        if "current_model_id" not in st.session_state:
            default_model_id, _ = get_default_model()
            return default_model_id
        return st.session_state.get("current_model_id")
    
    @staticmethod
    def sync_model_with_pipeline() -> None:
        """Sync the current model with the RAG pipeline."""
        SessionManager._update_last_activity()
        
        try:
            if st.session_state.rag_pipeline:
                current_model_id = st.session_state.get("current_model_id")
                if current_model_id:
                    st.session_state.rag_pipeline.set_model(current_model_id)
                    logger.info(f"Synced model with pipeline: {current_model_id}")
        except Exception as e:
            logger.warning(f"Failed to sync model with pipeline: {e}")
    
    # Message management methods
    @staticmethod
    def set_error_message(message: str) -> None:
        """Set error message."""
        SessionManager._update_last_activity()
        st.session_state.error_message = message
        logger.error(f"Error message set: {message}")
    
    @staticmethod
    def get_error_message() -> Optional[str]:
        """Get and clear error message."""
        SessionManager._update_last_activity()
        message = st.session_state.get("error_message")
        if message:
            st.session_state.error_message = None
        return message
    
    @staticmethod
    def set_success_message(message: str) -> None:
        """Set success message."""
        SessionManager._update_last_activity()
        st.session_state.success_message = message
        logger.info(f"Success message set: {message}")
    
    @staticmethod
    def get_success_message() -> Optional[str]:
        """Get and clear success message."""
        SessionManager._update_last_activity()
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
            
            # Reset TTL tracking
            st.session_state.session_created_at = time.time()
            st.session_state.last_activity = time.time()
            
            logger.info("Session reset successfully")
            
        except Exception as e:
            error_msg = f"Failed to reset session: {e}"
            logger.error(error_msg)
            st.error(error_msg)
    
    @staticmethod
    def get_session_info() -> Dict[str, Any]:
        """Get comprehensive session information."""
        SessionManager._update_last_activity()
        
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
        
        # Add TTL information
        ttl_info = SessionManager.get_session_ttl_info()
        info.update(ttl_info)
        
        if pipeline:
            try:
                pipeline_stats = pipeline.get_pipeline_stats()
                info["pipeline_stats"] = pipeline_stats
            except Exception as e:
                logger.warning(f"Could not get pipeline stats: {e}")
                info["pipeline_stats"] = {"error": str(e)}
        
        return info