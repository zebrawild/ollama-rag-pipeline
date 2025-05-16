"""Setup configuration for the RAG pipeline package."""

from setuptools import setup, find_packages

setup(
    name="ollama-rag-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "langchain",
        "langchain-ollama",
        "langchain-community",
        "langchain-unstructured",
        "python-multipart",
    ],
    python_requires=">=3.9",
) 