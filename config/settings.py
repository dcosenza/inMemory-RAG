"""Configuration settings for the RAG chatbot application."""

from typing import Dict, Any, List, Optional, Tuple
import os
from dataclasses import dataclass, field
from enum import Enum


class ModelProvider(Enum):
    """Supported model providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ModelInfo:
    """Information about a specific model."""
    name: str
    display_name: str
    provider: ModelProvider
    model_id: str
    max_tokens: int = 4000
    context_length: int = 8192
    is_free: bool = True
    description: str = ""
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for language models."""
    provider: ModelProvider = ModelProvider.OPENROUTER
    model_name: str = "openai/gpt-oss-20b:free"
    temperature: float = 0.7
    max_tokens: int = 1000
    streaming: bool = True
    
    def get_model_id(self) -> str:
        """Get the actual model ID for API calls."""
        return self.model_name
    
    def get_display_name(self) -> str:
        """Get the display name for UI."""
        return AVAILABLE_MODELS.get(self.model_name, {}).get("display_name", self.model_name)


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    similarity_threshold: float = 0.1
    max_results: int = 10
    metric_type: str = "METRIC_INNER_PRODUCT"


@dataclass
class DocumentConfig:
    """Configuration for document processing."""
    chunk_size: int = 1500
    chunk_overlap: int = 300
    max_file_size_mb: int = 10
    supported_formats: tuple = ("pdf",)


@dataclass
class AppConfig:
    """Main application configuration."""
    app_title: str = "RAG Document Chat"
    page_icon: str = "🤖"
    layout: str = "wide"
    max_memory_mb: int = 500
    session_ttl_minutes: int = 10  # Session timeout in minutes
    session_warning_minutes: int = 2  # Warning before session expires


# Available models configuration
AVAILABLE_MODELS = {
    # Free models
    "mistralai/mistral-7b-instruct:free": {
        "display_name": "Mistral 7B Instruct",
        "provider": ModelProvider.OPENROUTER,
        "max_tokens": 4000,
        "context_length": 8192,
        "is_free": True,
        "description": "Fast and efficient 7B parameter model",
        "capabilities": ["chat", "reasoning", "code"]
    },
    "openai/gpt-oss-20b:free": {
        "display_name": "GPT OSS 20B",
        "provider": ModelProvider.OPENROUTER,
        "max_tokens": 4000,
        "context_length": 8192,
        "is_free": True,
        "description": "Open source 20B parameter model",
        "capabilities": ["chat", "reasoning", "code"]
    },
    "z-ai/glm-4.5-air:free": {
        "display_name": "Z.AI GLM-4.5-AIR",
        "provider": ModelProvider.OPENROUTER,
        "max_tokens": 4000,
        "context_length": 32768,
        "is_free": True,
        "description": "High-performance mixture of experts model",
        "capabilities": ["chat", "reasoning", "code", "long-context"]
    },
    # Paid models (for future expansion)
    "openai/gpt-4o": {
        "display_name": "GPT-4o",
        "provider": ModelProvider.OPENROUTER,
        "max_tokens": 4096,
        "context_length": 128000,
        "is_free": False,
        "description": "Latest GPT-4 model with enhanced capabilities",
        "capabilities": ["chat", "reasoning", "code", "vision", "long-context"]
    },
    "anthropic/claude-3-5-sonnet": {
        "display_name": "Claude 3.5 Sonnet",
        "provider": ModelProvider.OPENROUTER,
        "max_tokens": 4096,
        "context_length": 200000,
        "is_free": False,
        "description": "Anthropic's latest Claude model",
        "capabilities": ["chat", "reasoning", "code", "long-context"]
    }
}

# Legacy FREE_MODELS for backward compatibility
FREE_MODELS = {
    model_info["display_name"]: model_id
    for model_id, model_info in AVAILABLE_MODELS.items()
    if model_info["is_free"]
}


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from .env file or environment variables."""
    api_key = ""
    
    # Try .env file first (for local development)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key and api_key != "your_api_key_here":
            return api_key
    except Exception:
        pass
    
    # Fallback to environment variable
    try:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key and api_key != "your_api_key_here":
            return api_key
    except Exception:
        pass
    
    # If we get here, no API key was found
    raise ValueError(
        "OpenRouter API key not found. Please set it in:\n"
        "1. Local: Create .env file with 'OPENROUTER_API_KEY=your_actual_key'\n"
        "2. Environment: Set OPENROUTER_API_KEY environment variable\n"
        "Get your free API key at: https://openrouter.ai/"
    )


def get_available_models(include_paid: bool = False) -> Dict[str, str]:
    """Get available models for selection."""
    if include_paid:
        return {
            model_info["display_name"]: model_id
            for model_id, model_info in AVAILABLE_MODELS.items()
        }
    else:
        return FREE_MODELS.copy()


def get_default_model() -> Tuple[str, str]:
    """Get the default model ID and display name from configuration."""
    default_model_id = MODEL_CONFIG.model_name
    default_display_name = AVAILABLE_MODELS.get(default_model_id, {}).get("display_name", default_model_id)
    return default_model_id, default_display_name


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a model."""
    return AVAILABLE_MODELS.get(model_id)


def validate_model(model_id: str) -> bool:
    """Validate if a model ID is supported."""
    return model_id in AVAILABLE_MODELS


def get_model_by_display_name(display_name: str) -> Optional[str]:
    """Get model ID by display name."""
    for model_id, model_info in AVAILABLE_MODELS.items():
        if model_info["display_name"] == display_name:
            return model_id
    return None


# API Configuration - Handle gracefully if not available at import time
try:
    OPENROUTER_API_KEY = get_openrouter_api_key()
except ValueError:
    # Don't fail at import time, let the app handle this gracefully
    OPENROUTER_API_KEY = ""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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