"""Tests for the QA chain functionality."""

import pytest
from unittest.mock import Mock, patch
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult

from ollama_rag_pipeline.chain import create_qa_chain, get_current_time_context

class DummyDoc:
    def __init__(self, content):
        self.page_content = content

class DummyLLM(BaseLLM):
    def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
        return LLMResult(generations=[[{"text": "Test answer"}]])
    def _llm_type(self):
        return "dummy"

@pytest.fixture
def mock_llm():
    llm = DummyLLM()
    return llm

@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    mock = Mock()
    mock.get_relevant_documents.return_value = [DummyDoc("Test context")]
    return mock

def test_get_current_time_context():
    """Test that current time context returns a properly formatted string."""
    time_str = get_current_time_context()
    assert isinstance(time_str, str)
    assert len(time_str) > 0

def test_create_qa_chain_has_correct_vars(mock_llm, mock_retriever):
    """Test that the QA chain's prompt has the correct input variables."""
    chain = create_qa_chain(mock_llm, mock_retriever)
    assert chain.prompt.input_variables == ["context", "current_datetime", "query"]

def test_qa_chain_run_returns_plain_string(mock_llm, mock_retriever):
    """Test that the QA chain returns a plain string."""
    chain = create_qa_chain(mock_llm, mock_retriever)
    result = chain.invoke({
        "query": "Test question",
        "context": "Test context",
        "current_datetime": "2024-01-01 00:00:00"
    })
    assert isinstance(result, str)
    assert result == "Test answer"
    mock_retriever.get_relevant_documents.assert_called_once()

def test_prompt_includes_current_datetime(mock_llm, mock_retriever):
    """Test that the prompt template correctly includes all required variables."""
    chain = create_qa_chain(mock_llm, mock_retriever)
    filled = chain.prompt.format(
        context="Test context",
        query="Test question",
        current_datetime="2024-01-01 00:00:00"
    )
    assert "2024-01-01 00:00:00" in filled
    assert "Test context" in filled
    assert "Test question" in filled

def test_qa_chain_error_handling(mock_llm, mock_retriever):
    """Test QA chain error handling."""
    mock_retriever.get_relevant_documents.side_effect = Exception("Retriever error")
    chain = create_qa_chain(mock_llm, mock_retriever)
    with pytest.raises(Exception):
        chain.invoke({
            "query": "Test question",
            "context": "Test context",
            "current_datetime": "2024-01-01 00:00:00"
        })

@patch("ollama_rag_pipeline.chain.get_current_time_context")
def test_qa_chain_with_mocked_time(mock_time, mock_llm, mock_retriever):
    """Test QA chain with mocked time context."""
    mock_time.return_value = "2024-03-20 12:00:00"
    
    chain = create_qa_chain(mock_llm, mock_retriever)
    result = chain.invoke({
        "query": "Test question",
        "context": "Test context",
        "current_datetime": "2024-03-20 12:00:00"
    })
    assert result == "Test answer" 