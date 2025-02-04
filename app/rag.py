from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List, Optional, Union, Dict
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel
import importlib
import os
import sys
import time
import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try both import styles
try:
    from app.config import MODEL_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, USE_LOCAL_EMBEDDINGS
    from app.templates import TEMPLATES
    from app.memory_manager import MemoryManager
    from app.embeddings import LocalEmbeddings
except ModuleNotFoundError:
    from config import MODEL_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, USE_LOCAL_EMBEDDINGS
    from templates import TEMPLATES
    from memory_manager import MemoryManager
    from embeddings import LocalEmbeddings

# Supported file extensions and their corresponding workflows
FILE_TYPE_MAP = {
    'image': ['png', 'jpg', 'jpeg', 'gif'],
    'document': ['pdf', 'txt', 'doc', 'docx'],
}

# Define a mapping of categories to workflows
WORKFLOW_MAP = {
    "General": ".general_workflow",  # Relative import
    "Multimodal": ".multimodal_workflow"  # Relative import
}

class FileInput(BaseModel):
    """Represents a file input with its content and metadata"""
    content: bytes
    file_type: str
    filename: str

def get_document_hash(file_path):
    """Generate a unique hash for the document."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_embeddings():
    """Get the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Ensure consistent normalization
    )

def process_file(file_path):
    """Process a file and create a vector store."""
    try:
        # Load and split the document
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            logger.error("No text chunks extracted from document")
            return False
            
        # Get embeddings
        embeddings = get_embeddings()
        
        # Create vector store
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store
        vector_store_path = file_path + "_faiss"
        vector_store.save_local(vector_store_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False

def load_vector_store(file_path):
    """Load a vector store for a file."""
    try:
        vector_store_path = file_path + "_faiss"
        if not os.path.exists(vector_store_path):
            logger.error(f"Vector store not found at {vector_store_path}")
            return None
            
        embeddings = get_embeddings()
        vector_store = FAISS.load_local(vector_store_path, embeddings)
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading vector store: {str(e)}")
        return None

def detect_multimodal_query(query: str) -> bool:
    """Detect if a query requires multimodal processing."""
    start_time = time.time()
    multimodal_keywords = [
        'image', 'picture', 'photo', 'document', 'file',
        'look at', 'analyze', 'show', 'read', 'scan'
    ]
    result = any(keyword in query.lower() for keyword in multimodal_keywords)
    logger.info(f"Query type detection completed in {time.time() - start_time:.2f}s")
    return result

def execute_workflow(memory, question: str, file_path: str = None):
    """Execute the appropriate workflow based on input type."""
    start_time = time.time()
    try:
        # Initialize LLM
        llm = ChatOpenAI(model=MODEL_NAME)
        logger.info("LLM initialized")
        
        if file_path:
            # Load document and get context
            vector_store = load_vector_store(file_path)
            if vector_store is None:
                logger.error("Failed to load vector store")
                return {
                    "answer": "Error processing request",
                    "chat_history": memory.load_memory_variables({}).get("chat_history", "")
                }
            
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create and execute chain
            chain = TEMPLATES["qa"] | llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "context": context,
                "chat_history": memory.load_memory_variables({}).get("chat_history", "")
            })
        else:
            # Simple chat without document context
            response = llm.invoke([{"role": "user", "content": question}]).content
        
        # Update memory
        memory.save_context(
            {"question": question},
            {"answer": response}
        )
        
        logger.info(f"Workflow completed in {time.time() - start_time:.2f}s")
        return {
            "answer": response,
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        return {
            "answer": f"Error processing request: {str(e)}",
            "chat_history": memory.load_memory_variables({}).get("chat_history", "")
        }

def main():
    """Main function for handling queries and file processing."""
    try:
        memory = MemoryManager()
        print("\nWelcome to the Study Assistant! Type 'quit' to exit.")
        
        while True:
            try:
                # Get user input
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() == 'quit':
                    print("\nGoodbye!")
                    break
                    
                # Check for file input
                file_path = input("Enter file path (or press Enter to skip): ").strip()
                file_input = None
                
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                        file_input = FileInput(
                            content=file_content,
                            file_type="document",
                            filename=file_path
                        )
                        process_file(file_path)
                
                # Execute workflow and get response
                response = execute_workflow(memory, question, file_path)
                print("\nResponse:", response["answer"])
                
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                continue
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
