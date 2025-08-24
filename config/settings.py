"""Configuration settings for the RAG chatbot application."""

from typing import Dict, Any
import os
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for language models."""
    provider: str = "openrouter"
    model_name: str = "mistralai/mistral-7b-instruct:free"
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    similarity_threshold: float = 0.7
    max_results: int = 5
    metric_type: str = "METRIC_INNER_PRODUCT"


@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 10
    supported_formats: tuple = ("pdf",)


@dataclass
class AppConfig:
    """Main application configuration."""
    app_title: str = "RAG Document Chat"
    page_icon: str = "🤖"
    layout: str = "wide"
    max_memory_mb: int = 500


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from Streamlit secrets or environment variables."""
    api_key = ""
    
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and 'OPENROUTER_API_KEY' in st.secrets:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            if api_key:
                return api_key
    except (ImportError, AttributeError, KeyError, FileNotFoundError):
        # Streamlit not available or secrets not configured
        pass
    
    # Fallback to environment variable (for local development)
    try:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key:
            return api_key
    except Exception:
        pass
    
    # If we get here, no API key was found
    raise ValueError(
        "OpenRouter API key not found. Please set it in:\n"
        "1. Streamlit Cloud: Go to Settings → Secrets → Add 'OPENROUTER_API_KEY = \"your_key\"'\n"
        "2. Local: Create .streamlit/secrets.toml with 'OPENROUTER_API_KEY = \"your_key\"'\n"
        "3. Environment: Set OPENROUTER_API_KEY environment variable"
    )

# API Configuration - Handle gracefully if not available at import time
try:
    OPENROUTER_API_KEY = get_openrouter_api_key()
except ValueError:
    # Don't fail at import time, let the app handle this gracefully
    OPENROUTER_API_KEY = ""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model options for OpenRouter free tier
FREE_MODELS = {
    "Mistral 7B Instruct": "mistralai/mistral-7b-instruct:free",
    "Meta Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct:free",
    "Mixtral 8x7B": "mistralai/mixtral-8x7b-instruct:free",
}

# Default configurations
MODEL_CONFIG = ModelConfig()
EMBEDDING_CONFIG = EmbeddingConfig()
VECTOR_STORE_CONFIG = VectorStoreConfig()
DOCUMENT_CONFIG = DocumentConfig()
APP_CONFIG = AppConfig()

# System prompt template
SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided document context. 

Guidelines:
- Answer questions using ONLY the information from the provided context
- If the context doesn't contain relevant information, say "I don't have enough information in the provided documents to answer that question"
- Be concise but comprehensive in your answers
- Always cite which document or section your answer comes from when possible
- If asked about something not in the context, politely redirect to document-based questions

Context:
{context}

Question: {question}

Answer:"""