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


# Environment variables
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
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