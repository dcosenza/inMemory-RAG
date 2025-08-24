"""Language model service for OpenRouter integration."""

from typing import Optional, Iterator, Dict, Any
import logging

from openai import OpenAI
from langchain.schema import Document

from config.settings import (
    MODEL_CONFIG, 
    OPENROUTER_API_KEY, 
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT,
    AVAILABLE_MODELS,
    get_available_models,
    get_model_info,
    validate_model,
    get_model_by_display_name
)
from core.exceptions import LLMError, ConfigurationError

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with language models via OpenRouter."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize LLM service with configuration."""
        self.config = config or MODEL_CONFIG
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize OpenRouter client."""
        try:
            # Get API key using the config function (handles Streamlit secrets + env vars)
            from config.settings import get_openrouter_api_key
            api_key = get_openrouter_api_key()
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL
            )
            logger.info("Initialized OpenRouter client")
        except ValueError as e:
            # Re-raise configuration errors with clear message
            raise ConfigurationError(str(e))
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenRouter client: {e}")
    
    def set_model(self, model_name: str) -> None:
        """Set the current model for the service."""
        if not validate_model(model_name):
            raise LLMError(f"Invalid model: {model_name}")
        
        self.config.model_name = model_name
        logger.info(f"Model set to: {model_name}")
    
    def set_model_by_display_name(self, display_name: str) -> None:
        """Set model by its display name."""
        model_id = get_model_by_display_name(display_name)
        if not model_id:
            raise LLMError(f"Model not found: {display_name}")
        
        self.set_model(model_id)
    
    def get_current_model(self) -> str:
        """Get the current model ID."""
        return self.config.model_name
    
    def get_current_model_display_name(self) -> str:
        """Get the current model display name."""
        model_info = get_model_info(self.config.model_name)
        return model_info.get("display_name", self.config.model_name) if model_info else self.config.model_name
    
    def _prepare_context(self, relevant_documents: list[Document]) -> str:
        """Prepare context from relevant documents."""
        if not relevant_documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(relevant_documents, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", "")
            
            context_parts.append(
                f"Document {i} (Source: {source}, Chunk: {chunk_id}):\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _create_messages(self, query: str, context: str) -> list[Dict[str, str]]:
        """Create messages for the chat completion."""
        system_message = SYSTEM_PROMPT.format(context=context, question=query)
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query}
        ]
    
    def generate_response(
        self, 
        query: str, 
        relevant_documents: list[Document],
        model_name: Optional[str] = None,
        stream: Optional[bool] = None
    ) -> str:
        """Generate a response using the specified model."""
        if not self.client:
            raise LLMError("LLM client not initialized")
        
        # Use provided model or current model
        if model_name:
            if not validate_model(model_name):
                raise LLMError(f"Invalid model: {model_name}")
        else:
            model_name = self.config.model_name
        
        should_stream = stream if stream is not None else self.config.streaming
        
        try:
            context = self._prepare_context(relevant_documents)
            messages = self._create_messages(query, context)
            
            if should_stream:
                return self._generate_streaming_response(messages, model_name)
            else:
                return self._generate_complete_response(messages, model_name)
                
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to generate response: {e}")
    
    def _generate_complete_response(self, messages: list, model_name: str) -> str:
        """Generate complete response (non-streaming)."""
        try:
            # Get model-specific configuration
            model_info = get_model_info(model_name)
            max_tokens = model_info.get("max_tokens", self.config.max_tokens) if model_info else self.config.max_tokens
            
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise LLMError(
                    "OpenRouter API authentication failed. Please check your API key:\n"
                    "1. Go to https://openrouter.ai/ and get a valid API key\n"
                    "2. Update your .env file\n"
                    "3. Restart the application"
                )
            else:
                raise LLMError(f"Failed to generate complete response: {e}")
    
    def _generate_streaming_response(self, messages: list, model_name: str) -> Iterator[str]:
        """Generate streaming response."""
        try:
            # Get model-specific configuration
            model_info = get_model_info(model_name)
            max_tokens = model_info.get("max_tokens", self.config.max_tokens) if model_info else self.config.max_tokens
            
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Unauthorized" in error_msg:
                raise LLMError(
                    "OpenRouter API authentication failed. Please check your API key:\n"
                    "1. Go to https://openrouter.ai/ and get a valid API key\n"
                    "2. Update your .env file\n"
                    "3. Restart the application"
                )
            else:
                raise LLMError(f"Failed to generate streaming response: {e}")
    
    def generate_streaming_response(
        self,
        query: str,
        relevant_documents: list[Document],
        model_name: Optional[str] = None
    ) -> Iterator[str]:
        """Generate streaming response for real-time display."""
        if not self.client:
            raise LLMError("LLM client not initialized")
        
        # Use provided model or current model
        if model_name:
            if not validate_model(model_name):
                raise LLMError(f"Invalid model: {model_name}")
        else:
            model_name = self.config.model_name
        
        try:
            context = self._prepare_context(relevant_documents)
            messages = self._create_messages(query, context)
            
            yield from self._generate_streaming_response(messages, model_name)
            
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to generate streaming response: {e}")
    
    def validate_model(self, model_name: str) -> bool:
        """Validate if a model is available."""
        return validate_model(model_name)
    
    def get_available_models(self, include_paid: bool = False) -> Dict[str, str]:
        """Get available models for selection."""
        return get_available_models(include_paid)
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        return get_model_info(model_name)
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        model_info = get_model_info(self.config.model_name)
        
        return {
            "current_model": self.config.model_name,
            "current_display_name": self.get_current_model_display_name(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "streaming": self.config.streaming,
            "available_models": list(get_available_models().keys()),
            "client_initialized": self.client is not None,
            "model_details": model_info or {}
        }