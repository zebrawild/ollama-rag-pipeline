"""Shared test fixtures."""

import pytest
from unittest.mock import Mock
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock(spec=OllamaLLM)
    llm.invoke.return_value = "Test answer"
    return llm

@pytest.fixture
def mock_embedding():
    """Create a mock embedding model for testing."""
    embedding = Mock(spec=OllamaEmbeddings)
    embedding.embed_query.return_value = [0.1, 0.2, 0.3]
    return embedding

@pytest.fixture
def mock_vectordb():
    """Create a mock vector database for testing."""
    vectordb = Mock(spec=Chroma)
    vectordb.as_retriever.return_value = Mock()
    return vectordb

@pytest.fixture
def mock_qa_chain():
    """Create a mock QA chain for testing."""
    chain = Mock()
    chain.invoke.return_value = {"answer": "Test answer"}
    return chain 