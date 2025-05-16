"""Data models for the RAG pipeline."""

from pydantic import BaseModel

class Query(BaseModel):
    """Query model for the question-answering endpoint."""
    question: str 