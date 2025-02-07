"""
Workflow management module
"""
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
from langchain.vectorstores import VectorStore
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from app.core.memory_manager import ConversationMemory
from app.config import MODEL_NAME, MODEL_TEMPERATURE
from app.workflows.Vision_workflow import VisionWorkflow
from app.loaders import DocumentLoaderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    GENERAL = "general"
    MULTIMODAL = "multimodal"
    VISION = "vision"

class BaseWorkflow:
    """Base class for all workflows"""
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            temperature=MODEL_TEMPERATURE
        )
    
    def handle_query(self, query: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

class GeneralWorkflow(BaseWorkflow):
    """Workflow for general queries without document context"""
    def handle_query(self, query: str, **kwargs) -> Dict[str, Any]:
        response = self.llm.invoke(query).content
        return {
            "answer_generator": response,
            "sources": []
        }

class MultimodalWorkflow(BaseWorkflow):
    """Workflow for document-based queries"""
    def __init__(self, memory: ConversationMemory, vector_store: VectorStore):
        super().__init__(memory)
        self.vector_store = vector_store
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document using appropriate loader"""
        logger.info(f"Loading document: {file_path}")
        
        if not DocumentLoaderFactory.is_supported_file(file_path):
            raise ValueError(f"Unsupported file type: {Path(file_path).suffix}")
        
        loader = DocumentLoaderFactory.get_loader(file_path)
        return loader.load()
    
    def handle_query(self, query: str, **kwargs) -> Dict[str, Any]:
        file_path = kwargs.get('file_path')
        if not file_path:
            raise ValueError("File path is required for multimodal workflow")
        
        # Load and process document
        try:
            documents = self.load_document(file_path)
            logger.info(f"Loaded {len(documents)} document chunks")
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(query)
            
            # Format documents for context
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # Get chat history if available
            chat_history = self.memory.get_chat_history() if self.memory else ""
            
            # Generate response with context and chat history
            prompt = f"""Based on the following document content:

{context}

Previous conversation:
{chat_history}

Question: {query}

Please provide a detailed answer based solely on the information from the document."""
            
            response = self.llm.invoke(prompt).content
            
            # Update memory with the interaction
            if self.memory:
                self.memory.add_interaction(query, response)
            
            return {
                "answer_generator": response,
                "sources": relevant_docs
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

class WorkflowManager:
    """Manages different types of workflows"""
    
    def __init__(self):
        """Initialize the workflow manager"""
        self.memory = ConversationMemory()
        self.vector_store = None
        self.current_workflow_type = WorkflowType.GENERAL
        # Initialize with general workflow
        self.active_workflow = GeneralWorkflow(self.memory)
        logger.info("WorkflowManager initialized with GeneralWorkflow")
    
    def set_vector_store(self, vector_store: VectorStore):
        """Set the vector store for document queries"""
        self.vector_store = vector_store
    
    def set_vision_workflow(self):
        """Set the workflow for image-based queries"""
        logger.info("Setting up VisionWorkflow with memory")
        self.active_workflow = VisionWorkflow(self.memory)
        self.current_workflow_type = WorkflowType.VISION
    
    def _determine_workflow_type(self, **kwargs) -> WorkflowType:
        """Determine which workflow to use based on the query context"""
        if "image_path" in kwargs:
            return WorkflowType.VISION
        elif self.vector_store is not None and "file_path" in kwargs:
            return WorkflowType.MULTIMODAL
        elif self.current_workflow_type != WorkflowType.GENERAL:
            # Maintain current workflow for follow-up queries
            return self.current_workflow_type
        else:
            return WorkflowType.GENERAL
    
    def _switch_workflow(self, workflow_type: WorkflowType):
        """Switch to a different workflow"""
        if workflow_type == self.current_workflow_type:
            return
        
        logger.info(f"Switching to {workflow_type.value} workflow")
        
        # Save current memory
        current_memory = self.memory
        
        if workflow_type == WorkflowType.GENERAL:
            self.active_workflow = GeneralWorkflow(current_memory)
        elif workflow_type == WorkflowType.MULTIMODAL:
            if self.vector_store is None:
                raise ValueError("Vector store is required for multimodal workflow")
            self.active_workflow = MultimodalWorkflow(current_memory, self.vector_store)
        elif workflow_type == WorkflowType.VISION:
            self.active_workflow = VisionWorkflow(current_memory)
        
        self.current_workflow_type = workflow_type
        logger.info(f"Successfully switched to {workflow_type.value} workflow with memory")
    
    def route_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Route the query to appropriate workflow"""
        try:
            # Determine workflow type
            workflow_type = self._determine_workflow_type(**kwargs)
            logger.info(f"Determined workflow type: {workflow_type.value}")
            
            # Switch workflow if needed
            self._switch_workflow(workflow_type)
            
            # Route query to active workflow
            logger.info(f"Routing query to {workflow_type.value} workflow")
            response = self.active_workflow.handle_query(query, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            raise

def test_query_routing():
    """Example method to test workflow routing with different scenarios"""
    logger.info("Starting Workflow Routing Test")
    
    # Initialize workflow manager
    workflow_manager = WorkflowManager()
    
    # Test 1: General Query
    logger.info("\n=== Test 1: General Query ===")
    query1 = "What is machine learning?"
    logger.info(f"Query: {query1}")
    response1 = workflow_manager.route_query(query1)
    logger.info(f"Workflow Type: {workflow_manager.current_workflow_type.value}")
    logger.info(f"Response: {response1['answer_generator']}")
    
    # Test 2: Vision Query
    logger.info("\n=== Test 2: Vision Query ===")
    from pathlib import Path
    test_image = Path(__file__).parent.parent.parent / "tests" / "test_data" / "test_image.jpg"
    
    if test_image.exists():
        query2 = "What objects do you see in this image?"
        logger.info(f"Query: {query2}")
        response2 = workflow_manager.route_query(query2, image_path=str(test_image))
        logger.info(f"Workflow Type: {workflow_manager.current_workflow_type.value}")
        logger.info(f"Response: {response2['answer_generator']}")
    else:
        logger.warning("Test image not found!")
    
    # Test 3: Document Query
    logger.info("\n=== Test 3: Document Query ===")
    from app.core.rag_engine import RAGEngine
    test_doc = Path(__file__).parent.parent.parent / "tests" / "test_data" / "sample.pdf"
    
    if test_doc.exists():
        logger.info(f"Processing document: {test_doc}")
        rag_engine = RAGEngine()
        doc_id = rag_engine.process_document(str(test_doc))
        workflow_manager.set_vector_store(rag_engine.get_vector_store(doc_id))
        
        query3 = "What does this document discuss?"
        logger.info(f"Query: {query3}")
        response3 = workflow_manager.route_query(query3)
        logger.info(f"Workflow Type: {workflow_manager.current_workflow_type.value}")
        logger.info(f"Response: {response3['answer_generator']}")
    else:
        logger.warning("Test document not found!")

if __name__ == "__main__":
    test_query_routing()
