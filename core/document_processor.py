"""Document processing module for PDF extraction and text chunking."""

from typing import List, Optional, Tuple
import io
import logging
from pathlib import Path

import PyPDF2
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import DOCUMENT_CONFIG
from core.exceptions import DocumentProcessingError, FileSizeError, UnsupportedFileFormatError

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Handles PDF document processing and text extraction."""
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize document processor with configuration."""
        self.config = config or DOCUMENT_CONFIG
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def validate_file(self, file_data: bytes, filename: str) -> None:
        """Validate uploaded file size and format."""
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise FileSizeError(
                f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed "
                f"size ({self.config.max_file_size_mb}MB)"
            )
        
        # Check file format
        file_extension = Path(filename).suffix.lower().lstrip('.')
        if file_extension not in self.config.supported_formats:
            raise UnsupportedFileFormatError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(self.config.supported_formats)}"
            )
    
    def extract_text_pypdf2(self, file_data: bytes) -> str:
        """Extract text using PyPDF2 (fallback method)."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_data))
            text_content = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_content)
        except Exception as e:
            raise DocumentProcessingError(f"PyPDF2 extraction failed: {e}")
    
    def extract_text_pdfplumber(self, file_data: bytes) -> str:
        """Extract text using pdfplumber (primary method)."""
        try:
            text_content = []
            
            with pdfplumber.open(io.BytesIO(file_data)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue
            
            return "\n\n".join(text_content)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
            return None
    
    def extract_text_from_pdf(self, file_data: bytes, filename: str) -> str:
        """Extract text from PDF using multiple extraction methods."""
        self.validate_file(file_data, filename)
        
        # Try pdfplumber first (better for complex layouts)
        extracted_text = self.extract_text_pdfplumber(file_data)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not extracted_text or not extracted_text.strip():
            logger.info("Falling back to PyPDF2 for text extraction")
            extracted_text = self.extract_text_pypdf2(file_data)
        
        if not extracted_text or not extracted_text.strip():
            raise DocumentProcessingError(
                f"Failed to extract text from {filename}. "
                "The PDF might be scanned or contain only images."
            )
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters from {filename}")
        return extracted_text
    
    def chunk_text(self, text: str, filename: str) -> List[Document]:
        """Split text into chunks and create Document objects."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "char_count": len(chunk)
                        }
                    )
                    documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {filename}")
            return documents
        
        except Exception as e:
            raise DocumentProcessingError(f"Failed to chunk text from {filename}: {e}")
    
    def process_pdf(self, file_data: bytes, filename: str) -> List[Document]:
        """Complete pipeline: extract text and create document chunks."""
        try:
            # Extract text
            text = self.extract_text_from_pdf(file_data, filename)
            
            # Create chunks
            documents = self.chunk_text(text, filename)
            
            if not documents:
                raise DocumentProcessingError(f"No valid content found in {filename}")
            
            return documents
        
        except Exception as e:
            if isinstance(e, (DocumentProcessingError, FileSizeError, UnsupportedFileFormatError)):
                raise
            else:
                raise DocumentProcessingError(f"Unexpected error processing {filename}: {e}")
    
    def get_processing_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents."""
        if not documents:
            return {}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get("source", "Unknown") for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(documents),
            "unique_sources": len(sources),
            "source_files": list(sources)
        }