"""Document processing and vector store management."""

import logging
import os
import tempfile
import shutil
from typing import List, Optional, Union, Tuple, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    TextLoader,
    CSVLoader,
    PythonLoader,
    JSONLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader
)
from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_document(path2document):
    logger.info(f"Loading local document: {path2document}")
    ext = os.path.splitext(path2document)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path2document)
    elif ext == ".txt":
        loader = TextLoader(path2document)
    elif ext == ".csv":
        loader = CSVLoader(path2document)
    elif ext == ".py":
        loader = PythonLoader(path2document)
    elif ext == ".json":
        loader = JSONLoader(path2document)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(path2document)
    elif ext in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(path2document)
    elif ext in [".pptx", ".ppt"]:
        loader = UnstructuredPowerPointLoader(path2document)
    elif ext in [".html", ".htm"]:
        loader = UnstructuredHTMLLoader(path2document)
    else:
        loader = UnstructuredFileLoader(path2document)

    return loader.load()


def _process_metadata(metadata: dict) -> dict:
    """Process metadata to handle lists and other complex types."""
    processed = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            processed[key] = ", ".join(str(item) for item in value)
        else:
            processed[key] = value
    return processed

def _split_markdown_document(doc: Document) -> List[Document]:
    """Split a markdown document into chunks while preserving headers."""
    # Define headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    
    # Create markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Split the document
    header_splits = markdown_splitter.split_text(doc.page_content)
    
    # Further split long sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    
    # Process each split
    final_splits = []
    for split in header_splits:
        # Add header information to metadata
        metadata = {**doc.metadata, **split.metadata}
        
        # Further split if needed
        if len(split.page_content) > Config.CHUNK_SIZE:
            sub_splits = text_splitter.split_text(split.page_content)
            for sub_split in sub_splits:
                final_splits.append(Document(
                    page_content=sub_split,
                    metadata=metadata
                ))
        else:
            final_splits.append(Document(
                page_content=split.page_content,
                metadata=metadata
            ))
    
    return final_splits

def convert_to_document(doc: Union[str,Tuple[str, dict], Document, None]) -> Optional[Document]:
    """Convert various input types to a Document object."""
    if doc is None:
        return None
        
    try:
        # If it's already a Document, process its metadata
        if isinstance(doc, Document):
            processed_metadata = _process_metadata(doc.metadata)
            return Document(page_content=doc.page_content, metadata=processed_metadata)
            
        # If it's a tuple of (content, metadata)
        if isinstance(doc, tuple) and len(doc) == 2:
            content, metadata = doc
            if not isinstance(content, str):
                logger.warning(f"Content must be string, got {type(content)}")
                return None
            if metadata is not None and not isinstance(metadata, dict):
                logger.warning(f"Metadata must be dict, got {type(metadata)}")
                return None
            processed_metadata = _process_metadata(metadata or {})
            return Document(page_content=content, metadata=processed_metadata)
        #path to a file 
        if isinstance(doc, str):
            return load_document(doc)                   
        
        logger.warning(f"Unexpected document type: {type(doc)}")
        return None
        
    except Exception as e:
        logger.error(f"Error converting document: {e}")
        return None

def process_documents(docs: List[Union[Document, Tuple[str, dict], None]], embedding) -> Optional[Chroma]:
    """Process documents and create/update vector store."""
    if not docs:
        logger.warning("No documents to process")
        return None
        
    try:
        # Convert all documents to Document objects
        converted_docs = []
        for i, doc in enumerate(docs, 1):
            converted = convert_to_document(doc)
            if converted:
                converted_docs.append(converted)
            else:
                logger.warning(f"Skipping document {i} due to conversion error")
        
        if not converted_docs:
            logger.warning("No valid documents after conversion")
            return None
            
        # Process documents
        all_chunks = []
        for doc in converted_docs:
            # Handle both string paths and Document objects
            if isinstance(doc.page_content, str) and doc.page_content.strip().startswith('#'):
                # If it's a markdown document, use markdown splitter
                chunks = _split_markdown_document(doc)
            else:
                # For other documents, use regular text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=Config.CHUNK_SIZE,
                    chunk_overlap=Config.CHUNK_OVERLAP
                )
                chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        
        if not all_chunks:
            logger.warning("No chunks created after splitting")
            return None
            
        logger.info(f"Creating vector store with {len(all_chunks)} chunks")
        # Create or update vector store
        try:
            # Create a temporary directory for Chroma
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory for vector store: {temp_dir}")
            
            # Create vector store in temporary directory
            vectordb = Chroma.from_documents(
                documents=all_chunks,
                embedding=embedding,
                persist_directory=temp_dir
            )
            vectordb.persist()
            logger.info("Vector store created and persisted successfully")
            
            # Move the contents to the final location
            if os.path.exists(Config.VECTOR_DB_DIR):
                shutil.rmtree(Config.VECTOR_DB_DIR)
            shutil.move(temp_dir, Config.VECTOR_DB_DIR)
            logger.info(f"Moved vector store to {Config.VECTOR_DB_DIR}")
            
            # Ensure proper permissions
            os.chmod(Config.VECTOR_DB_DIR, 0o777)
            for root, dirs, files in os.walk(Config.VECTOR_DB_DIR):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o666)
            
            # Reload the vector store from the final location
            vectordb = Chroma(
                persist_directory=Config.VECTOR_DB_DIR,
                embedding_function=embedding
            )
            return vectordb
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            # Clean up temporary directory if it exists
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return None
        
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return None

def initialize_vectordb(embedding) -> Optional[Chroma]:
    """Initialize vector database from documents in docs directory."""
    try:
        # Try to load existing vector store
        if os.path.exists(Config.VECTOR_DB_DIR):
            logger.info("Loading existing vector database...")
            return Chroma(
                persist_directory=Config.VECTOR_DB_DIR,
                embedding_function=embedding
            )
            
        # If no existing vector store, create new one
        if not os.path.isdir(Config.DOCS_DIR):
            logger.warning(f"Docs directory not found: {Config.DOCS_DIR}")
            return None
            
        logger.info("Creating new vector database...")
        # Load all documents from the docs directory
        docs = []
        for filename in os.listdir(Config.DOCS_DIR):
            if filename.endswith(Config.SUPPORTED_FILE_TYPES):
                file_path = os.path.join(Config.DOCS_DIR, filename)
                loader = UnstructuredLoader(file_path)
                docs.extend(loader.load())
                
        if not docs:
            logger.warning("No documents found in docs directory")
            return None
            
        # Process documents and create vector store
        vectordb = process_documents(docs, embedding)
        if vectordb:
            vectordb.persist()
            logger.info("Vector database created and persisted")
        return vectordb
        
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        return None 
