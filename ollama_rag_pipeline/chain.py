"""QA chain creation and time context management."""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence

from .config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_current_time_context() -> str:
    """Get current time context for the QA chain."""
    now = datetime.now()
    weekday = now.strftime("%A")  # Full weekday name
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Add meal timing context
    hour = now.hour
    if 5 <= hour < 11:
        meal_context = "breakfast time"
    elif 11 <= hour < 15:
        meal_context = "lunch time"
    elif 15 <= hour < 19:
        meal_context = "dinner time"
    else:
        meal_context = "evening"
    
    return f"Current date and time: {weekday}, {date} at {time} ({meal_context})"

class QAChain:
    """Wrapper class for the QA chain that handles retrieval and time context."""
    
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever):
        """Initialize the QA chain with LLM and retriever."""
        self.llm = llm
        self.retriever = retriever
        
        # Create prompt template
        template = """You are a helpful AI assistant. Use the following pieces of context and current time to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Current time: {current_datetime}

        Context: {context}

        Question: {query}

        Answer:"""

        self.prompt = PromptTemplate(
            input_variables=["context", "current_datetime", "query"],
            template=template
        )

        # Compose the chain using the new RunnableSequence API
        self.chain = self.prompt | self.llm

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the chain with the given inputs."""
        try:
            # Get relevant documents from retriever
            docs = self.retriever.get_relevant_documents(inputs["query"])
            context = "\n".join(doc.page_content for doc in docs)
            
            # Log retrieved context
            logger.info("Retrieved context for query '%s':", inputs["query"])
            for i, doc in enumerate(docs, 1):
                logger.info("Document %d:", i)
                logger.info("Content: %s", doc.page_content)
                if doc.metadata:
                    logger.info("Metadata: %s", doc.metadata)
                logger.info("---")
            
            # Add context to inputs
            chain_inputs = {
                "query": inputs["query"],
                "context": context,
                "current_datetime": inputs.get("current_datetime", get_current_time_context())
            }
            
            # Call the chain
            return self.chain.invoke(chain_inputs)
        except Exception as e:
            logger.error(f"Error in QA chain invocation: {e}")
            raise

def create_qa_chain(llm: BaseLLM, retriever: BaseRetriever) -> QAChain:
    """Create a QA chain with the given LLM and retriever."""
    try:
        return QAChain(llm, retriever)
    except Exception as e:
        logger.error(f"Error creating QA chain: {e}")
        raise 