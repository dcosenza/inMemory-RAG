"""Main Streamlit application for RAG Document Chat."""

import logging
import sys
from typing import List, Tuple

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import modules after logging configuration
from config.settings import FREE_MODELS
from core.exceptions import RAGChatbotError, ConfigurationError
from utils.session_manager import SessionManager
from ui.components import UIComponents

logger = logging.getLogger(__name__)


def process_documents(files_data: List[Tuple[bytes, str]]) -> None:
    """Process uploaded documents."""
    if not files_data:
        return
    
    try:
        rag_pipeline = SessionManager.get_rag_pipeline()
        
        # Show processing indicator
        with st.spinner("🔄 Processing documents..."):
            processing_stats = rag_pipeline.process_documents(files_data)
        
        # Update session state
        SessionManager.set_documents_processed(True, processing_stats)
        
        # Show success message
        files_count = processing_stats.get('files_processed', 0)
        chunks_count = processing_stats.get('documents_created', 0)
        
        UIComponents.show_success(
            f"Successfully processed {files_count} files into {chunks_count} searchable chunks!"
        )
        
        # Show any errors
        if processing_stats.get('files_failed', 0) > 0:
            UIComponents.show_warning(
                f"Failed to process {processing_stats['files_failed']} files. Check the details above."
            )
    
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error(error_msg)
        UIComponents.show_error(error_msg)
        SessionManager.set_error_message(error_msg)


def handle_chat_query(user_input: str, selected_model: str) -> None:
    """Handle user chat query."""
    if not user_input:
        return
    
    try:
        rag_pipeline = SessionManager.get_rag_pipeline()
        model_name = FREE_MODELS.get(selected_model)
        
        if not model_name:
            UIComponents.show_error(f"Invalid model selected: {selected_model}")
            return
        
        # Add user message to chat history
        SessionManager.add_chat_message("user", user_input)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                # Get streaming response
                response_stream, sources, scores = rag_pipeline.query_streaming(
                    user_input,
                    model_name=model_name
                )
                
                # Render streaming response
                full_response = UIComponents.render_streaming_response(
                    response_stream, sources, scores
                )
                
                # Prepare sources for chat history
                source_info = []
                for doc, score in zip(sources, scores):
                    source_info.append({
                        "filename": doc.metadata.get("source", "Unknown"),
                        "content": doc.page_content[:300],  # First 300 chars
                        "score": score,
                        "chunk_id": doc.metadata.get("chunk_id", 0)
                    })
                
                # Add assistant message to chat history
                SessionManager.add_chat_message("assistant", full_response, source_info)
                
            except Exception as e:
                error_msg = f"Failed to generate response: {str(e)}"
                logger.error(error_msg)
                UIComponents.show_error(error_msg)
                
                # Add error message to chat history
                SessionManager.add_chat_message("assistant", "I'm sorry, I encountered an error while processing your question. Please try again.")
    
    except Exception as e:
        error_msg = f"Chat query failed: {str(e)}"
        logger.error(error_msg)
        UIComponents.show_error(error_msg)


def check_configuration() -> bool:
    """Check if the application is properly configured."""
    try:
        # Check if OpenRouter API key is set
        import os
        if not os.getenv("OPENROUTER_API_KEY"):
            UIComponents.show_error(
                "OpenRouter API key is not configured. Please set the OPENROUTER_API_KEY environment variable."
            )
            return False
        
        # Try to initialize RAG pipeline
        try:
            rag_pipeline = SessionManager.get_rag_pipeline()
            validation = rag_pipeline.validate_configuration()
            
            if not validation["overall"]:
                UIComponents.show_error(
                    f"Configuration validation failed: {validation}"
                )
                return False
            
            return True
            
        except ConfigurationError as e:
            UIComponents.show_error(f"Configuration error: {str(e)}")
            return False
    
    except Exception as e:
        UIComponents.show_error(f"Unexpected configuration error: {str(e)}")
        return False


def main():
    """Main application function."""
    try:
        # Initialize session
        SessionManager.initialize_session()
        
        # Render header
        UIComponents.render_header()
        
        # Check configuration
        if not check_configuration():
            st.stop()
        
        # Render sidebar and get user inputs
        files_data, selected_model = UIComponents.render_sidebar()
        
        # Show any pending error/success messages
        error_msg = SessionManager.get_error_message()
        if error_msg:
            UIComponents.show_error(error_msg)
        
        success_msg = SessionManager.get_success_message()
        if success_msg:
            UIComponents.show_success(success_msg)
        
        # Main content area
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Process documents if uploaded
            if files_data and not SessionManager.is_documents_processed():
                process_documents(files_data)
                st.rerun()
            
            # Show document processing status
            if files_data or SessionManager.is_documents_processed():
                processing_stats = SessionManager.get_processing_stats()
                UIComponents.render_document_processing_status(files_data or [], processing_stats)
                st.divider()
            
            # Chat interface
            user_input = UIComponents.render_chat_interface()
            
            # Handle chat query
            if user_input:
                handle_chat_query(user_input, selected_model)
                st.rerun()
        
        with col2:
            # Show helpful tips
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Upload multiple PDFs for comprehensive answers
            - Ask specific questions for better results
            - Try different AI models for varied responses
            - Use the sources to verify information
            """)
            
            if SessionManager.is_documents_processed():
                st.markdown("### ❓ Example Questions")
                st.markdown("""
                - "What are the main topics covered?"
                - "Can you summarize the key findings?"
                - "What conclusions can be drawn?"
                - "Are there any recommendations?"
                """)
        
        # Render footer
        UIComponents.render_footer()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        UIComponents.show_error(f"Application error: {str(e)}")
        
        # Provide recovery options
        st.markdown("### 🔧 Recovery Options")
        if st.button("🔄 Restart Application"):
            SessionManager.reset_session()
            st.rerun()


if __name__ == "__main__":
    main()