# RAG Document Chat

A **hybrid document analysis tool** that processes your PDF documents locally and uses AI to answer questions about them. Upload your documents, ask questions in natural language, and get intelligent answers backed by your actual content - with local document processing and cloud-based AI responses.

## Why would you use it

### **Enhanced Privacy**
- **Your documents are processed locally** - text extraction, chunking, and embeddings happen on your machine
- **No persistent storage** - documents are deleted when you close the app
- **Minimal data exposure** - only relevant document chunks are sent to AI services

### **Intelligent Analysis**
- **Ask anything** about your documents in plain English
- **Get precise answers** with source references
- **Real-time responses** with streaming for immediate feedback
- **Multi-document support** - analyze multiple PDFs at once

### **Key Features**

| Feature | What It Does | Why It Matters |
|---------|-------------|----------------|
| **Local Document Processing** | PDFs processed entirely on your machine | Your documents never leave your control during processing |
| **Source Attribution** | Shows exactly which parts of documents were used | Verify answers and dive deeper into specific sections |
| **Multiple AI Models** | Choose from various free models | Find the best AI personality for your needs |
| **Session Isolation** | Each session is completely separate | Perfect for sharing with others without data mixing |
| **Streaming Responses** | See answers appear in real-time | More engaging experience, no waiting for complete responses |
| **Smart Chunking** | Breaks documents into optimal pieces | Better context understanding and more accurate answers |

## Technology & Architecture

### **Core Stack**

| Technology | What It Does |
|------------|-------------|
| **Streamlit** | Web interface framework |
| **LangChain** | RAG pipeline orchestration |
| **FAISS** | Vector similarity search |
| **Sentence Transformers** | Text embeddings |
| **OpenRouter** | AI model access |
| **PyPDF2 + pdfplumber** | PDF text extraction |

### **Why These Choices**

- **LangChain**: Provides the proven RAG architecture that powers enterprise applications
- **FAISS**: Enables lightning-fast similarity search even with thousands of document chunks
- **Local Embeddings**: No API costs, no data sent to external services, instant processing
- **OpenRouter**: Access to multiple AI models without vendor lock-in
- **Dual PDF Processing**: Ensures we can extract text from even the most complex PDFs

## How It Works

```
Document Upload
    ↓
PDF Processing & Validation (LOCAL)
    ↓
Text Extraction (LOCAL - pdfplumber → PyPDF2 fallback)
    ↓
Smart Chunking (LOCAL - 1500 chars with 300 overlap)
    ↓
Local Embedding Generation (LOCAL - sentence-transformers)
    ↓
FAISS Vector Storage (LOCAL - in-memory)
    ↓
User Query
    ↓
Query Embedding + Similarity Search (LOCAL)
    ↓
Top-K Relevant Chunks Retrieved (LOCAL)
    ↓
AI Response Generation (CLOUD - OpenRouter API)
    ↓
Source Attribution + Streaming Display
    ↓
Automatic Cleanup (LOCAL - session end)
```

### **Data Flow Explanation**

1. **Local Document Processing**: PDFs are validated, extracted, and chunked into digestible pieces on your machine
2. **Local Vector Creation**: Each chunk gets converted to a numerical vector (embedding) locally
3. **Local Storage**: Vectors are stored in FAISS for ultra-fast similarity search on your machine
4. **Local Query Processing**: Your question gets converted to a vector and compared to all chunks locally
5. **Local Retrieval**: Most relevant document chunks are found locally
6. **Cloud Response**: Relevant chunks and your question are sent to OpenRouter for AI response generation
7. **Display**: Answer streams back with source references
8. **Local Cleanup**: Everything is automatically deleted from your machine when you're done

## Quick Start (Docker)

### Prerequisites
- **Docker** installed on your machine
- **OpenRouter API key** (free at https://openrouter.ai/)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/inMemory-RAG.git
   cd inMemory-RAG
   ```

2. **Get your API key**
   - Go to https://openrouter.ai/
   - Sign up for free account
   - Copy your API key

3. **Set your API key**
   ```bash
   export OPENROUTER_API_KEY="your_actual_api_key_here"
   ```

4. **Run with Docker**
   ```bash
   # Build and start the application
   docker-compose up --build
   ```

5. **Access the app**
   - Open your browser
   - Go to: **http://localhost:8501**
   - Start uploading PDFs and chatting!