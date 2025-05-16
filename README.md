# Ollama RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline built with FastAPI and Ollama, designed to provide context-aware responses by combining document retrieval with language model generation.

## Features

- Document ingestion and processing with automatic chunking
- Vector database storage using Chroma
- FastAPI-based REST API
- Integration with Ollama for embeddings and language model
- Support for markdown documents with header preservation
- Automatic time context awareness for temporal queries

## Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- Required Ollama models:
  - `nomic-embed-text` for embeddings
  - `mistral` for language model

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ollama-rag-pipeline.git
cd ollama-rag-pipeline
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Pull required Ollama models:
```bash
ollama pull nomic-embed-text
ollama pull mistral
```

## Configuration

The application can be configured through environment variables or by modifying `ollama_rag_pipeline/config.py`. Key configuration options include:

- `HOST`: API server host (default: "127.0.0.1")
- `PORT`: API server port (default: 8000)
- `DOCS_DIR`: Directory containing documents to process
- `VECTOR_DB_DIR`: Directory for vector database storage
- `CHUNK_SIZE`: Size of document chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `EMBEDDING_MODEL`: Ollama model for embeddings (default: "nomic-embed-text")
- `LLM_MODEL`: Ollama model for language model (default: "mistral")

## Running the Application

1. Start the FastAPI server:
```bash
uvicorn ollama_rag_pipeline.rag_pipeline:app --reload
```

The server will be available at `http://127.0.0.1:8000`

## API Endpoints

### POST /query
Submit a query to the RAG pipeline.

Request body:
```json
{
    "query": "Your question here"
}
```

### POST /upload
Upload a new document to be processed and added to the vector database.

Form data:
- `file`: The document file to upload (supported formats: .md, .txt)

### POST /clear
Clear the vector database and reinitialize it from the documents directory.

## Document Processing

The pipeline processes documents with the following features:

- Automatic chunking with configurable size and overlap
- Markdown header preservation
- Metadata extraction and preservation
- Support for multiple document formats

## Development

### Project Structure
```
ollama_rag_pipeline/
├── __init__.py
├── chain.py          # QA chain implementation
├── config.py         # Configuration settings
├── document_processor.py  # Document processing logic
└── rag_pipeline.py   # FastAPI application
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 