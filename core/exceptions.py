"""Custom exceptions for the RAG chatbot application."""


class RAGChatbotError(Exception):
    """Base exception for RAG chatbot application."""
    pass


class DocumentProcessingError(RAGChatbotError):
    """Exception raised during document processing."""
    pass


class EmbeddingError(RAGChatbotError):
    """Exception raised during embedding generation."""
    pass


class VectorStoreError(RAGChatbotError):
    """Exception raised during vector store operations."""
    pass


class LLMError(RAGChatbotError):
    """Exception raised during LLM operations."""
    pass


class ConfigurationError(RAGChatbotError):
    """Exception raised for configuration errors."""
    pass


class FileSizeError(DocumentProcessingError):
    """Exception raised when file size exceeds limits."""
    pass


class UnsupportedFileFormatError(DocumentProcessingError):
    """Exception raised for unsupported file formats."""
    pass