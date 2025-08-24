"""UI components for the Streamlit RAG chatbot application."""

from typing import List, Dict, Any, Optional, Tuple
import logging

import streamlit as st
from langchain.schema import Document

from config.settings import get_available_models, get_model_info, AVAILABLE_MODELS, APP_CONFIG
from utils.session_manager import SessionManager

logger = logging.getLogger(__name__)


class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def render_header() -> None:
        """Render application header."""
        st.set_page_config(
            page_title=APP_CONFIG.app_title,
            page_icon=APP_CONFIG.page_icon,
            layout=APP_CONFIG.layout,
            initial_sidebar_state="expanded"
        )
        
        st.title(f"{APP_CONFIG.page_icon} {APP_CONFIG.app_title}")
        st.markdown("""
        Upload PDF documents and chat with them using AI. Your documents are processed privately 
        and deleted when your session ends.
        """)
        st.divider()
    
    @staticmethod
    def render_sidebar() -> Tuple[Optional[List[Tuple[bytes, str]]], str]:
        """Render sidebar with file upload and model selection."""
        with st.sidebar:
            st.header("📄 Document Upload")
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to chat with"
            )
            
            # Convert uploaded files to the format expected by the pipeline
            files_data = None
            if uploaded_files:
                files_data = []
                for file in uploaded_files:
                    file_bytes = file.read()
                    files_data.append((file_bytes, file.name))
            
            st.divider()
            
            # Model selection
            st.header("🤖 Model Settings")
            UIComponents._render_model_selection()
            
            st.divider()
            
            # Session controls
            st.header("⚙️ Session Controls")
            
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                SessionManager.clear_chat_history()
                st.rerun()
            
            if st.button("🔄 Reset Session", use_container_width=True):
                SessionManager.reset_session()
                st.rerun()
            
            # Session info (collapsible)
            with st.expander("📊 Session Info"):
                UIComponents._render_session_info()
            
            return files_data, SessionManager.get_current_model_id()
    
    @staticmethod
    def _render_model_selection() -> None:
        """Render model selection interface."""
        try:
            # Get available models
            available_models = get_available_models()
            current_display_name = SessionManager.get_current_model_display_name()
            
            # Ensure current model is in available models (fallback to first if not)
            if current_display_name not in available_models:
                logger.warning(f"Current model '{current_display_name}' not in available models, using first available")
                current_display_name = list(available_models.keys())[0]
                SessionManager.set_current_model(current_display_name)
            
            # Model selection dropdown
            selected_display_name = st.selectbox(
                "Choose AI Model",
                options=list(available_models.keys()),
                index=list(available_models.keys()).index(current_display_name),
                help="Select the AI model for generating responses"
            )
            
            # Handle model change
            if selected_display_name != current_display_name:
                try:
                    SessionManager.set_current_model(selected_display_name)
                    st.success(f"✅ Switched to {selected_display_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to switch model: {str(e)}")
            
            # Show current model info
            current_model_id = SessionManager.get_current_model_id()
            model_info = get_model_info(current_model_id)
            
            if model_info:
                st.markdown("**Current Model Info:**")
                
                # Model capabilities
                capabilities = model_info.get("capabilities", [])
                if capabilities:
                    st.markdown(f"**Capabilities:** {', '.join(capabilities)}")
                
                # Model description
                description = model_info.get("description", "")
                if description:
                    st.markdown(f"**Description:** {description}")
                
                # Model stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Tokens", model_info.get("max_tokens", "Unknown"))
                with col2:
                    st.metric("Context Length", f"{model_info.get('context_length', 0):,}")
                
                # Free/Paid indicator
                is_free = model_info.get("is_free", True)
                status = "🆓 Free" if is_free else "💰 Paid"
                st.markdown(f"**Status:** {status}")
            
        except Exception as e:
            st.error(f"❌ Error loading model selection: {str(e)}")
            logger.error(f"Model selection error: {e}")
    
    @staticmethod
    def _render_session_info() -> None:
        """Render session information in sidebar."""
        try:
            session_info = SessionManager.get_session_info()
            
            # Basic info
            st.write(f"**Documents Loaded:** {'✅' if session_info['documents_processed'] else '❌'}")
            st.write(f"**Chat Messages:** {session_info['chat_history_length']}")
            st.write(f"**Current Model:** {session_info['current_model_display_name']}")
            
            # Processing stats
            if session_info.get('processing_stats'):
                stats = session_info['processing_stats']
                if stats.get('documents_created'):
                    st.write(f"**Document Chunks:** {stats['documents_created']}")
                if stats.get('memory_usage_mb'):
                    st.write(f"**Memory Usage:** {stats['memory_usage_mb']:.1f} MB")
            
            # Pipeline stats
            if session_info.get('pipeline_stats'):
                pipeline_stats = session_info['pipeline_stats']
                if pipeline_stats.get('available_models_count'):
                    st.write(f"**Available Models:** {pipeline_stats['available_models_count']}")
        
        except Exception as e:
            st.write("⚠️ Could not load session info")
            logger.warning(f"Failed to render session info: {e}")
    
    @staticmethod
    def render_document_processing_status(files_data: List[Tuple[bytes, str]], processing_stats: Dict[str, Any]) -> None:
        """Render document processing status."""
        if not files_data:
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if processing_stats:
                if processing_stats.get('files_processed', 0) > 0:
                    st.success(f"✅ Successfully processed {processing_stats['files_processed']} files")
                    
                    if processing_stats.get('documents_created'):
                        st.info(f"📝 Created {processing_stats['documents_created']} document chunks")
                    
                    if processing_stats.get('memory_usage_mb'):
                        st.info(f"💾 Using {processing_stats['memory_usage_mb']:.1f} MB of memory")
                
                if processing_stats.get('files_failed', 0) > 0:
                    st.warning(f"⚠️ Failed to process {processing_stats['files_failed']} files")
                    
                    for error in processing_stats.get('errors', []):
                        st.error(f"❌ {error}")
            else:
                st.info("📤 Ready to process uploaded documents")
        
        with col2:
            file_count = len(files_data)
            total_size = sum(len(file_data) for file_data, _ in files_data) / (1024 * 1024)
            st.metric("Files", file_count)
            st.metric("Total Size", f"{total_size:.1f} MB")
    
    @staticmethod
    def render_chat_interface() -> Optional[str]:
        """Render chat interface and return user input."""
        # Display chat history
        UIComponents._render_chat_history()
        
        # Chat input
        if not SessionManager.is_documents_processed():
            st.info("👆 Please upload and process some PDF documents first to start chatting!")
            return None
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_area(
                    "Ask a question about your documents:",
                    placeholder="What is the main topic discussed in the documents?",
                    height=100,
                    label_visibility="collapsed"
                )
            
            with col2:
                st.write("")  # Spacing
                submitted = st.form_submit_button("Send 📤", use_container_width=True)
        
        return user_input.strip() if submitted and user_input.strip() else None
    
    @staticmethod
    def _render_chat_history() -> None:
        """Render chat history."""
        chat_history = SessionManager.get_chat_history()
        
        if not chat_history:
            st.info("💬 No chat history yet. Ask your first question!")
            return
        
        # Create scrollable chat container
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                
                elif message["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                        
                        # Show sources if available
                        if message.get("sources"):
                            UIComponents._render_sources(message["sources"])
    
    @staticmethod
    def _render_sources(sources: List[Dict[str, Any]]) -> None:
        """Render source documents."""
        if not sources:
            return
        
        with st.expander(f"📚 Sources ({len(sources)} documents)", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {source.get('filename', 'Unknown')}")
                st.markdown(f"*Relevance Score:* {source.get('score', 0):.3f}")
                
                # Show snippet of content
                content = source.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                st.markdown(f"*Preview:* {content}")
                
                if i < len(sources):  # Don't add divider after last source
                    st.divider()
    
    @staticmethod
    def render_streaming_response(response_stream, sources: List[Document], scores: List[float]) -> str:
        """Render streaming response from the assistant."""
        # Create placeholder for streaming content
        response_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in response_stream:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")  # Show cursor
        
        # Final response without cursor
        response_placeholder.markdown(full_response)
        
        # Prepare sources for storage
        source_info = []
        for doc, score in zip(sources, scores):
            source_info.append({
                "filename": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content[:500],  # First 500 chars
                "score": score,
                "chunk_id": doc.metadata.get("chunk_id", 0)
            })
        
        # Show sources
        if source_info:
            UIComponents._render_sources(source_info)
        
        return full_response
    
    @staticmethod
    def show_error(message: str) -> None:
        """Show error message."""
        st.error(f"❌ {message}")
    
    @staticmethod
    def show_success(message: str) -> None:
        """Show success message."""
        st.success(f"✅ {message}")
    
    @staticmethod
    def show_info(message: str) -> None:
        """Show info message."""
        st.info(f"ℹ️ {message}")
    
    @staticmethod
    def show_warning(message: str) -> None:
        """Show warning message."""
        st.warning(f"⚠️ {message}")
    
    @staticmethod
    def render_footer() -> None:
        """Render application footer."""
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(
                """
                <div style='text-align: center; color: #666; padding: 20px;'>
                    <p>🔒 Your documents are processed privately and automatically deleted when your session ends.</p>
                    <p>Powered by OpenRouter • Built with Streamlit</p>
                </div>
                """,
                unsafe_allow_html=True
            )