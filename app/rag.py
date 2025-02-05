import logging
import os
import pickle
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
from pydantic import BaseModel

import hashlib
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try both import styles
try:
    from app.config import MODEL_NAME, EMBEDDING_MODEL, USE_LOCAL_EMBEDDINGS
    from app.templates import TEMPLATES
    from app.embeddings import LocalEmbeddings
except ModuleNotFoundError:
    from config import MODEL_NAME, EMBEDDING_MODEL, USE_LOCAL_EMBEDDINGS
    from templates import TEMPLATES
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
    """Input schema for file processing."""
    file_path: str

class VectorStoreOutput(BaseModel):
    """Output schema for vector store operations."""
    vector_store_path: str

def get_document_hash(file_path):
    """Generate a unique hash for the document."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_embeddings():
    """Get the embedding model."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Ensure consistent normalization
        )
        return embeddings
    except Exception as e:
        logger.error(f"Error initializing embeddings: {e}")
        raise

def process_file(file_path: str) -> Optional[str]:
    """
    Process a PDF file and create embeddings.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Path to the vector store if successful, None otherwise
    """
    try:
        file_input = FileInput(file_path=file_path)
        logger.debug(f"Starting file processing")
        logger.debug(f"File path: {file_input.file_path}")
        logger.debug(f"File exists: {os.path.exists(file_input.file_path)}")
        
        # Validate file
        if not os.path.exists(file_input.file_path):
            logger.error(f"File does not exist: {file_input.file_path}")
            return None
        
        # Load PDF
        loader = PyMuPDFLoader(file_input.file_path)
        documents = loader.load()
        logger.debug(f"Loaded {len(documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        logger.debug(f"Created {len(splits)} text chunks")
        
        # Validate splits
        if not splits:
            logger.error("No document splits created")
            return None
        
        # Get embeddings
        embeddings = get_embeddings()
        
        # Create vector store directory
        vector_store_path = Path("/Users/rayengallas/Desktop/Coding_projects/Study-Assistant-Clean/vector_store").absolute() / Path(file_input.file_path).stem
        vector_store_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Vector store directory: {vector_store_path}")
        
        # Create and persist vector store
        try:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=str(vector_store_path)
            )
            vector_store.persist()
            logger.debug("Vector store created and persisted successfully")
            
            result = VectorStoreOutput(vector_store_path=str(vector_store_path))
            return result.vector_store_path
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            return None
        
    except Exception as e:
        logger.error(f"Unexpected error in process_file: {e}", exc_info=True)
        return None

def load_vector_store(file_path: str) -> Optional[Chroma]:
    """
    Load the vector store from disk.
    
    Args:
        file_path (str): Path to the vector store directory
        
    Returns:
        Optional[Chroma]: Loaded vector store if successful, None otherwise
    """
    try:
        logger.debug(f"Loading vector store from: {file_path}")
        
        # Validate path
        if not os.path.exists(file_path):
            logger.error(f"Vector store directory not found: {file_path}")
            return None
            
        # Initialize embeddings
        embeddings = get_embeddings()
        
        # Load vector store
        vector_store = Chroma(
            persist_directory=file_path,
            embedding_function=embeddings
        )
        
        logger.debug("Vector store loaded successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error loading vector store: {e}", exc_info=True)
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

def execute_workflow(question: str, file_path: str = None):
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
                    "chat_history": ""
                }
            
            retriever = vector_store.as_retriever()
            docs = retriever.get_relevant_documents(question)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Create and execute chain
            chain = TEMPLATES["qa"] | llm | StrOutputParser()
            response = chain.invoke({
                "question": question,
                "context": context,
                "chat_history": ""
            })
        else:
            # Simple chat without document context
            response = llm.invoke([{"role": "user", "content": question}]).content
        
        logger.info(f"Workflow completed in {time.time() - start_time:.2f}s")
        return {
            "answer": response,
            "chat_history": ""
        }
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        return {
            "answer": f"Error processing request: {str(e)}",
            "chat_history": ""
        }

def main():
    """Main function for handling queries and file processing."""
    try:
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
                            file_path=file_path
                        )
                        vector_store_path = process_file(file_path)
                
                # Execute workflow and get response
                if file_path and os.path.exists(file_path):
                    response = execute_workflow(question, vector_store_path)
                else:
                    response = execute_workflow(question)
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

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_DIR = Path("/Users/rayengallas/Desktop/Coding_projects/Study-Assistant-Clean/vector_store").absolute()

if __name__ == "__main__":
    exit(main())
