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
    FREE_MODELS
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
        if not OPENROUTER_API_KEY:
            raise ConfigurationError(
                "OPENROUTER_API_KEY environment variable is required"
            )
        
        try:
            self.client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL
            )
            logger.info("Initialized OpenRouter client")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenRouter client: {e}")
    
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
        
        model_name = model_name or self.config.model_name
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
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise LLMError(f"Failed to generate complete response: {e}")
    
    def _generate_streaming_response(self, messages: list, model_name: str) -> Iterator[str]:
        """Generate streaming response."""
        try:
            stream = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
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
        
        model_name = model_name or self.config.model_name
        
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
        return model_name in FREE_MODELS.values()
    
    def get_available_models(self) -> Dict[str, str]:
        """Get available free models."""
        return FREE_MODELS.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model configuration."""
        return {
            "current_model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "streaming": self.config.streaming,
            "available_models": list(FREE_MODELS.keys()),
            "client_initialized": self.client is not None
        }