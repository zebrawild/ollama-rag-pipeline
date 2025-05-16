"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from langchain_community.vectorstores import Chroma

from ollama_rag_pipeline.rag_pipeline import app
from ollama_rag_pipeline.chain import create_qa_chain
from ollama_rag_pipeline.document_processor import process_documents

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)

@pytest.fixture
def mock_qa_chain():
    """Create a mock QA chain."""
    chain = Mock()
    chain.invoke.return_value = "Test answer"
    return chain

def test_query_endpoint_success(client):
    """Test successful query endpoint."""
    with patch("ollama_rag_pipeline.rag_pipeline.qa_chain") as mock_qa_chain, \
         patch("ollama_rag_pipeline.rag_pipeline.vectordb") as mock_vectordb:
        mock_qa_chain.invoke.return_value = "Test answer"
        mock_vectordb.similarity_search.return_value = [Mock(page_content="Relevant context")]
        response = client.post("/query", json={"query": "Test question"})
        assert response.status_code == 200
        assert response.json() == {"answer": "Test answer"}

def test_query_endpoint_no_chain(client):
    """Test query endpoint when QA chain is not initialized."""
    with patch("ollama_rag_pipeline.rag_pipeline.qa_chain", None):
        response = client.post("/query", json={"query": "Test question"})
        assert response.status_code == 503
        assert "RAG pipeline not initialized" in response.json()["detail"]

def test_query_endpoint_invalid_input(client, mock_qa_chain):
    """Test query endpoint with invalid input."""
    with patch("ollama_rag_pipeline.rag_pipeline.qa_chain", mock_qa_chain):
        # Test with empty query
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

        # Test with missing query
        response = client.post("/query", json={})
        assert response.status_code == 422

        # Test with invalid JSON
        response = client.post("/query", data="invalid json")
        assert response.status_code == 422

def test_upload_endpoint_success(client):
    """Test successful file upload."""
    with patch("ollama_rag_pipeline.rag_pipeline.vectordb") as mock_vectordb, \
         patch("ollama_rag_pipeline.rag_pipeline.qa_chain") as mock_qa_chain, \
         patch("ollama_rag_pipeline.rag_pipeline.process_documents") as mock_process, \
         patch("ollama_rag_pipeline.rag_pipeline.UnstructuredLoader") as mock_loader:
        # Setup mocks
        mock_loader.return_value.load.return_value = [
            {"page_content": "Test content", "metadata": {}}
        ]
        mock_process.return_value = Mock(spec=Chroma)
        # Create test file
        test_file = ("test.txt", b"Test content", "text/plain")
        response = client.post(
            "/upload",
            files={"file": test_file}
        )
        assert response.status_code == 200
        assert response.json() == {"message": "Document processed successfully"}
        mock_process.assert_called_once()

def test_upload_endpoint_invalid_file_type(client):
    """Test file upload with invalid file type."""
    test_file = ("test.xyz", b"Test content", "text/plain")
    response = client.post(
        "/upload",
        files={"file": test_file}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_endpoint_empty_file(client):
    """Test file upload with empty file."""
    with patch("ollama_rag_pipeline.rag_pipeline.UnstructuredLoader") as mock_loader:
        mock_loader.return_value.load.return_value = []
        
        test_file = ("test.txt", b"", "text/plain")
        response = client.post(
            "/upload",
            files={"file": test_file}
        )
        assert response.status_code == 400
        assert "No content extracted" in response.json()["detail"]

def test_upload_endpoint_processing_error(client):
    """Test file upload with processing error."""
    with patch("ollama_rag_pipeline.rag_pipeline.UnstructuredLoader") as mock_loader:
        mock_loader.return_value.load.side_effect = Exception("Processing error")
        
        test_file = ("test.txt", b"Test content", "text/plain")
        response = client.post(
            "/upload",
            files={"file": test_file}
        )
        assert response.status_code == 500
        assert "Failed to process" in response.json()["detail"] 