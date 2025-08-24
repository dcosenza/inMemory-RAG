# RAG Document Chat 🤖📄

A modern, privacy-focused RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and have intelligent conversations with their content. Built with Streamlit and deployed on Streamlit Cloud.

## ✨ Features

- **Private Document Processing**: Documents are processed in-memory and automatically deleted when sessions end
- **Multiple AI Models**: Choose from various free models via OpenRouter (Mistral, Llama, Mixtral)
- **Real-time Streaming**: Streaming responses for immediate feedback
- **Source Attribution**: See which parts of your documents generated each answer
- **Multi-document Support**: Upload and query multiple PDFs simultaneously
- **Modern UI**: Clean, responsive interface with chat history
- **Session Isolation**: Complete user privacy with isolated sessions

## 🏗️ Architecture

The application follows a modular, enterprise-grade architecture:

```
📁 rag-document-chat/
├── 📁 config/
│   └── settings.py          # Configuration management
├── 📁 core/
│   ├── document_processor.py   # PDF processing & chunking
│   ├── embedding_service.py    # Text embeddings (HuggingFace)
│   ├── vector_store.py         # FAISS vector storage
│   ├── llm_service.py          # OpenRouter LLM integration
│   ├── rag_pipeline.py         # Main RAG orchestration
│   └── exceptions.py           # Custom exceptions
├── 📁 utils/
│   └── session_manager.py      # Streamlit session management
├── 📁 ui/
│   └── components.py           # Reusable UI components
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Key Components

- **Document Processor**: Handles PDF text extraction using PyPDF2 and pdfplumber with fallback mechanisms
- **Embedding Service**: Uses sentence-transformers for local, free embeddings
- **Vector Store**: FAISS-based similarity search with configurable thresholds
- **LLM Service**: OpenRouter integration with streaming support
- **RAG Pipeline**: Orchestrates the entire retrieval-augmented generation process
- **Session Manager**: Ensures complete user isolation and privacy

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key (free tier available)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-document-chat.git
   cd rag-document-chat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Create a new app and select this repository
4. Add your `OPENROUTER_API_KEY` in the Streamlit Cloud secrets management
5. Deploy!

## 🔧 Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `LOG_LEVEL`: Logging level (optional, default: INFO)

### Model Configuration

The application supports multiple free models through OpenRouter:

- **Mistral 7B Instruct**: Fast, efficient model
- **Meta Llama 3.1 8B**: Balanced performance
- **Mixtral 8x7B**: High-quality responses

### Customization

Key settings can be modified in `config/settings.py`:

```python
# Document processing
chunk_size = 1000
chunk_overlap = 200
max_file_size_mb = 10

# Vector search
similarity_threshold = 0.7
max_results = 5

# Model settings
temperature = 0.7
max_tokens = 1000
```

## 🔒 Privacy & Security

- **Ephemeral Storage**: All documents and embeddings exist only in memory
- **Session Isolation**: Each user session is completely isolated
- **No Persistence**: No data is saved to disk or databases
- **Automatic Cleanup**: All data is automatically deleted when sessions end
- **Local Processing**: Embeddings are generated locally, not sent to external services

## 📋 Usage

1. **Upload Documents**: Use the sidebar to upload one or more PDF files
2. **Wait for Processing**: Documents are automatically processed and chunked
3. **Start Chatting**: Ask questions about your documents in natural language
4. **View Sources**: Expand source sections to see which documents informed each answer
5. **Switch Models**: Try different AI models for varied response styles

### Example Queries

- "What are the main topics discussed in these documents?"
- "Can you summarize the key findings?"
- "What recommendations are mentioned?"
- "Are there any specific statistics or numbers mentioned?"

## 🛠️ Development

### Project Structure

- **Separation of Concerns**: Clear separation between UI, business logic, and data layers
- **Dependency Injection**: Configurable components for easy testing
- **Error Handling**: Comprehensive exception handling with user-friendly messages
- **Logging**: Structured logging for debugging and monitoring
- **Type Hints**: Full type annotations for better code reliability

### Adding New Features

1. **New Document Types**: Extend `DocumentProcessor` class
2. **New Embedding Models**: Modify `EmbeddingService` configuration
3. **New LLM Providers**: Extend `LLMService` class
4. **UI Enhancements**: Add components to `ui/components.py`

### Testing

```bash
# Install development dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=core --cov-report=html
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure `OPENROUTER_API_KEY` is set correctly
2. **PDF Processing Fails**: Some PDFs may be image-based or corrupted
3. **Memory Issues**: Large documents may exceed memory limits
4. **Model Errors**: Check OpenRouter status and model availability

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

### Performance Optimization

- **Reduce chunk size** for large documents
- **Lower similarity threshold** for more results
- **Use smaller embedding models** for faster processing

## 📚 Technical Details

### RAG Pipeline Flow

1. **Document Upload** → PDF validation and size checking
2. **Text Extraction** → Multi-method extraction (pdfplumber → PyPDF2 fallback)
3. **Text Chunking** → Recursive character splitting with overlap
4. **Embedding Generation** → Local sentence-transformer embeddings
5. **Vector Storage** → FAISS index creation and storage
6. **Query Processing** → Query embedding and similarity search
7. **Context Retrieval** → Top-K relevant document chunks
8. **Response Generation** → LLM response with streaming
9. **Source Attribution** → Document source tracking and display

### Memory Management

- In-memory FAISS index for fast similarity search
- Automatic cleanup on session end
- Configurable memory limits and chunk sizes
- Efficient streaming to minimize memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [OpenRouter](https://openrouter.ai/) for free AI model access
- [LangChain](https://langchain.com/) for RAG pipeline components
- [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search
- [HuggingFace](https://huggingface.co/) for free embedding models

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/rag-document-chat/issues) page
2. Create a new issue with detailed information
3. Join our [Discord community](https://discord.gg/yourserver) for real-time help

---

**Built with ❤️ for the open-source community**