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
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
    from app.config import MODEL_NAME, EMBEDDING_MODEL
    from app.templates import TEMPLATES
    from app.memory_manager import MemoryManager
except ModuleNotFoundError:
    from config import MODEL_NAME, EMBEDDING_MODEL
    from templates import TEMPLATES
    from memory_manager import MemoryManager

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

# Constants for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create a persistent directory for the vector store
PERSIST_DIR = Path(__file__).parent / "vector_store"
PERSIST_DIR.mkdir(exist_ok=True)

def get_document_hash(file_path):
    """Generate a unique hash for the document."""
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def process_file(file_path):
    """Process a file and store its embeddings."""
    try:
        logger.info(f"Starting file processing for {file_path}")
        start_time = time.time()

        # Generate unique ID for this document
        doc_id = get_document_hash(file_path)
        persist_dir = PERSIST_DIR / doc_id

        # Check if we already have embeddings
        if persist_dir.exists():
            logger.info("Found existing embeddings, skipping processing")
            return True

        # Load the document
        logger.info("Loading document...")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Document loaded with {len(documents)} pages")

        # Split into chunks
        logger.info("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Create embeddings in batches
        logger.info("Creating embeddings...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Create vector store with persistence
        vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=str(persist_dir)
        )
        
        logger.info(f"File processing completed in {time.time() - start_time:.2f}s")
        return True

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        if persist_dir.exists():
            import shutil
            shutil.rmtree(persist_dir)  # Clean up failed processing
        raise

def load_and_split_documents(file_path):
    """Load and split a document into chunks for retrieval."""
    logger.info(f"Loading document from {file_path}")
    
    # Get document store path
    doc_id = get_document_hash(file_path)
    persist_dir = PERSIST_DIR / doc_id
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Load from existing store if available
    if persist_dir.exists():
        logger.info("Loading existing vector store")
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings
        ).as_retriever()
    
    # Otherwise, create new embeddings
    logger.info("Creating new vector store")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create and persist vector store
    vector_store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=str(persist_dir)
    )
    
    return vector_store.as_retriever()

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
            retriever = load_and_split_documents(file_path)
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
