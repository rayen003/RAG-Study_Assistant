"""
Base workflow class that other workflows inherit from
"""
import logging
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

from app.core.memory_manager import ConversationMemory
from app.templates import BASE_CONVERSATION_TEMPLATE

logger = logging.getLogger(__name__)

class BaseWorkflow:
    """Base class for all workflows"""
    
    def __init__(self, memory: ConversationMemory):
        """Initialize workflow with memory manager"""
        logger.info(f"Initializing {self.__class__.__name__}")
        self.memory = memory
        self.llm = ChatOpenAI(model="gpt-4o")
        self.prompt_template = self._create_base_template()
        self.chain = self._create_llm_chain()
    
    def _create_base_template(self) -> ChatPromptTemplate:
        """Create base template for the workflow"""
        return ChatPromptTemplate.from_messages([
            ("system", BASE_CONVERSATION_TEMPLATE),
            ("human", "{query}"),
            ("ai", "{previous_response}")
        ])
    
    def _create_llm_chain(self) -> LLMChain:
        """Create LLM chain with the template"""
        return LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
    
    def handle_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Handle a query - to be implemented by child classes"""
        raise NotImplementedError("Subclasses must implement handle_query")
