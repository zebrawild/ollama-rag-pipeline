"""RAG Pipeline with FastAPI integration."""

import logging
import os
import tempfile
import shutil
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_unstructured import UnstructuredLoader
from pydantic import BaseModel, Field

from .config import Config
from .chain import create_qa_chain, get_current_time_context
from .document_processor import process_documents, initialize_vectordb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
vectordb: Optional[Chroma] = None
qa_chain: Optional[OllamaLLM] = None

class Query(BaseModel):
    """Query model for question-answering."""
    query: str = Field(..., min_length=1, description="The question to answer")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    global vectordb, qa_chain
    
    # Startup
    logger.info("Initializing RAG pipeline...")
    embedding = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
    vectordb = initialize_vectordb(embedding)
    
    if vectordb:
        llm = OllamaLLM(model=Config.LLM_MODEL,stop=["[control_", "[control_36]", "] [control_"])
        qa_chain = create_qa_chain(llm, vectordb.as_retriever())
        logger.info("RAG pipeline initialized successfully")
    else:
        logger.warning("No documents found in docs directory")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG pipeline...")
    if vectordb:
        vectordb = None
    if qa_chain:
        qa_chain = None

# Initialize FastAPI app
app = FastAPI(title="RAG Pipeline API", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_endpoint(query: Query):
    """Process a query using the RAG pipeline."""
    if qa_chain is None or vectordb is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
    try:
        # Get current time context
        current_time = get_current_time_context()
        
        # Process query using the QA chain directly
        result = qa_chain.invoke({
            "query": query.query,
            "current_datetime": current_time
        })
        
        return {"answer": result}
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    if not file.filename.endswith(Config.SUPPORTED_FILE_TYPES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported types: {Config.SUPPORTED_FILE_TYPES}"
        )
        
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="No content extracted from file")
            temp_file.write(content)
            temp_path = temp_file.name
            logger.info(f"temp file located at {temp_path}")
            logger.info(f"temp file content {content}") 
            
        # Process document
        global vectordb, qa_chain
        embedding = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        vectordb = process_documents([temp_path], embedding)
        
        if vectordb is None:
            raise HTTPException(status_code=500, detail="Failed to process document")
            
        # Initialize QA chain
        llm = OllamaLLM(model=Config.LLM_MODEL)
        qa_chain = create_qa_chain(llm, vectordb.as_retriever())
        
        return {"message": "Document processed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
    finally:
        # Clean up temporary file
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {e}")

@app.post("/clear")
async def clear_vectordb():
    """Clear the vector database and reinitialize it."""
    global vectordb, qa_chain
    
    try:
        # Clear the vector database directory
        if os.path.exists(Config.VECTOR_DB_DIR):
            shutil.rmtree(Config.VECTOR_DB_DIR)
            logger.info("Vector database directory cleared")
        
        # Ensure the directory exists with proper permissions
        os.makedirs(Config.VECTOR_DB_DIR, exist_ok=True)
        os.chmod(Config.VECTOR_DB_DIR, 0o777)  # Full permissions for development
        logger.info("Vector database directory recreated with proper permissions")
        
        # Create new vector database from scratch
        embedding = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        
        # Load all documents from the docs directory
        docs = []
        logger.info(f"Scanning docs directory: {Config.DOCS_DIR}")
        for filename in os.listdir(Config.DOCS_DIR):
            if filename.endswith(Config.SUPPORTED_FILE_TYPES):
                file_path = os.path.join(Config.DOCS_DIR, filename)
                logger.info(f"Loading document: {file_path}")
                try:
                    loader = UnstructuredLoader(file_path)
                    loaded_docs = loader.load()
                    logger.info(f"Successfully loaded {len(loaded_docs)} chunks from {filename}")
                    docs.extend(loaded_docs)
                except Exception as e:
                    logger.error(f"Error loading document {filename}: {e}")
                    continue
                
        if not docs:
            logger.error(f"No documents found in {Config.DOCS_DIR}")
            raise HTTPException(status_code=500, detail="No documents found in docs directory")
            
        logger.info(f"Processing {len(docs)} total document chunks")
        # Process documents and create vector store
        vectordb = process_documents(docs, embedding)
        
        if vectordb:
            # Ensure all files in the vector database directory are writable
            for root, dirs, files in os.walk(Config.VECTOR_DB_DIR):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o666)
            
            llm = OllamaLLM(model=Config.LLM_MODEL,stop=["[control_", "[control_36]", "] [control_"])
            qa_chain = create_qa_chain(llm, vectordb.as_retriever())
            logger.info("Vector database reinitialized successfully")
            return {"message": "Vector database cleared and reinitialized successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reinitialize vector database")
            
    except Exception as e:
        logger.error(f"Error clearing vector database: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing vector database: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)
