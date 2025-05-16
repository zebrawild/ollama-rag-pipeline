"""Configuration settings for the RAG pipeline."""

import os

class Config:
    """Configuration constants for the RAG pipeline."""
    
    # Model Configuration
    EMBEDDING_MODEL       = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    LLM_MODEL             = os.getenv("LLM_MODEL", "gemma3:12b")
    
    # Document Processing
    CHUNK_SIZE            = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP         = int(os.getenv("CHUNK_OVERLAP", "200"))
    SUPPORTED_FILE_TYPES  = (".txt", ".pdf", ".md", ".docx")
    
    # Directory Configuration
    DOCS_DIR              = os.getenv("DOCS_DIR", "docs")
    VECTOR_DB_DIR         = os.getenv("VECTOR_DB_DIR", "./vector_db")
    
    # API Configuration
    HOST                  = os.getenv("HOST", "0.0.0.0")
    PORT                  = int(os.getenv("PORT", "8000"))
    
    VERBOSE = False 