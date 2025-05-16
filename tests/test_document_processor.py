"""Tests for document processing functionality."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings

from ollama_rag_pipeline.document_processor import convert_to_document, process_documents, initialize_vectordb

@pytest.fixture
def mock_embedding():
    """Create a mock embedding model for testing."""
    embedding = Mock(spec=OllamaEmbeddings)
    embedding.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedding

def test_convert_to_document_from_tuple():
    """Test converting a tuple to a Document."""
    # Test with simple metadata
    content = "Test content"
    metadata = {"source": "test.txt", "page": 1}
    doc = convert_to_document((content, metadata))
    assert isinstance(doc, Document)
    assert doc.page_content == content
    assert doc.metadata == metadata

    # Test with list metadata
    metadata_with_list = {"tags": ["test", "example"], "numbers": [1, 2, 3]}
    doc = convert_to_document((content, metadata_with_list))
    assert isinstance(doc, Document)
    assert doc.page_content == content
    assert doc.metadata["tags"] == "test, example"
    assert doc.metadata["numbers"] == "1, 2, 3"

    # Test with None metadata
    doc = convert_to_document((content, None))
    assert isinstance(doc, Document)
    assert doc.page_content == content
    assert doc.metadata == {}

def test_convert_to_document_from_document():
    """Test converting a Document to a Document (should process metadata)."""
    # Test with simple metadata
    content = "Test content"
    metadata = {"source": "test.txt", "page": 1}
    original_doc = Document(page_content=content, metadata=metadata)
    doc = convert_to_document(original_doc)
    assert isinstance(doc, Document)
    assert doc.page_content == content
    assert doc.metadata == metadata

    # Test with list metadata
    metadata_with_list = {"tags": ["test", "example"], "numbers": [1, 2, 3]}
    original_doc = Document(page_content=content, metadata=metadata_with_list)
    doc = convert_to_document(original_doc)
    assert isinstance(doc, Document)
    assert doc.page_content == content
    assert doc.metadata["tags"] == "test, example"
    assert doc.metadata["numbers"] == "1, 2, 3"

def test_convert_to_document_invalid_input():
    """Test converting invalid input to Document."""
    # Test with None
    assert convert_to_document(None) is None

    # Test with invalid tuple
    assert convert_to_document(("content",)) is None

    # Test with invalid type
    assert convert_to_document(123) is None

@patch("ollama_rag_pipeline.document_processor.RecursiveCharacterTextSplitter")
@patch("ollama_rag_pipeline.document_processor.Chroma")
def test_process_documents(mock_chroma, mock_splitter, mock_embedding):
    """Test document processing with mock splitter."""
    # Setup mock splitter
    mock_splitter.return_value.split_documents.return_value = [
        Document(page_content="chunk1", metadata={}),
        Document(page_content="chunk2", metadata={})
    ]
    mock_chroma.from_documents.return_value = Mock()
    # Test with valid documents
    docs = [
        Document(page_content="doc1", metadata={"source": "test1.txt"}),
        Document(page_content="doc2", metadata={"source": "test2.txt"})
    ]
    result = process_documents(docs, mock_embedding)
    assert result is not None
    mock_splitter.return_value.split_documents.assert_called_once()

    # Test with empty document list
    result = process_documents([], mock_embedding)
    assert result is None

    # Test with invalid documents
    result = process_documents([None, "invalid"], mock_embedding)
    assert result is None

@patch("ollama_rag_pipeline.document_processor.os")
@patch("ollama_rag_pipeline.document_processor.UnstructuredLoader")
@patch("ollama_rag_pipeline.document_processor.RecursiveCharacterTextSplitter")
@patch("ollama_rag_pipeline.document_processor.Chroma")
def test_initialize_vectordb(mock_chroma, mock_splitter, mock_loader, mock_os, mock_embedding):
    """Test vector database initialization."""
    # Setup mocks
    mock_os.path.isdir.return_value = True
    mock_os.listdir.return_value = ["test1.txt", "test2.pdf"]
    mock_loader.return_value.load.return_value = [
        Document(page_content="doc1", metadata={}),
        Document(page_content="doc2", metadata={})
    ]
    mock_splitter.return_value.split_documents.return_value = [
        Document(page_content="chunk1", metadata={}),
        Document(page_content="chunk2", metadata={})
    ]
    mock_chroma.from_documents.return_value = Mock()
    # Test successful initialization
    result = initialize_vectordb(mock_embedding)
    assert result is not None
    assert mock_loader.call_count == 2

    # Test with empty directory
    mock_os.listdir.return_value = []
    result = initialize_vectordb(mock_embedding)
    assert result is None

    # Test with non-existent directory
    mock_os.path.isdir.return_value = False
    result = initialize_vectordb(mock_embedding)
    assert result is None 