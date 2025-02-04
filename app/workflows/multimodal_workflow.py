from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.string import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import hashlib
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from app.templates import TEMPLATES
    from app.config import MODEL_NAME, EMBEDDING_MODEL
except ModuleNotFoundError:
    from ..templates import TEMPLATES
    from ..config import MODEL_NAME, EMBEDDING_MODEL

# Constants for text splitting
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Create a persistent directory for the vector store
# This is where we'll cache our document embeddings to avoid regenerating them
PERSIST_DIR = Path(__file__).parent.parent / "vector_store"
PERSIST_DIR.mkdir(exist_ok=True)

def get_document_hash(file_path):
    """
    Generate a unique hash for the document to use as cache key.
    This helps us identify if we've processed this exact document before.
    """
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_and_split_documents(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Load a PDF document and split it into chunks for processing.
    Now includes caching to avoid regenerating embeddings for the same document.
    
    Performance Optimizations:
    1. Caches document embeddings using Chroma's persistence
    2. Only generates embeddings once per unique document
    3. Uses batch processing for embedding creation
    """
    try:
        # OPTIMIZATION 1: Check cache first
        # Generate unique ID for this document
        doc_id = get_document_hash(file_path)
        persist_dir = PERSIST_DIR / doc_id
        
        # OPTIMIZATION 2: Reuse existing embeddings if available
        if persist_dir.exists():
            start_time = time.time()
            logger.info("Found cached embeddings, loading from disk...")
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
            vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=embeddings
            )
            logger.info(f"Loaded cached embeddings in {time.time() - start_time:.2f}s")
            return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # If not in cache, process the document
        logger.info("No cached embeddings found, processing document...")
        start_time = time.time()
        
        # OPTIMIZATION 3: Efficient document loading
        # Load the document in chunks to handle large files
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Document loaded in {time.time() - start_time:.2f}s")

        # OPTIMIZATION 4: Smart text splitting
        # Use RecursiveCharacterTextSplitter for better chunk boundaries
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]  # Try natural breaks first
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunks)} chunks")
        
        # OPTIMIZATION 5: Batch embedding creation
        # Create embeddings for all chunks in one batch operation
        embedding_start = time.time()
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma.from_documents(
            chunks, 
            embeddings,
            persist_directory=str(persist_dir)
        )
        vector_store.persist()  # Save to disk for future use
        logger.info(f"Embeddings created and cached in {time.time() - embedding_start:.2f}s")
        
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def multimodal_workflow(memory, question, file_path):
    """
    Process and answer questions about a document using LangChain.
    
    Performance Optimizations:
    1. Reuses cached embeddings
    2. Combines API calls where possible
    3. Better error handling and logging
    """
    start_time = time.time()
    try:
        # OPTIMIZATION 1: Single LLM instance
        # Create one LLM instance to reuse
        llm = ChatOpenAI(model=MODEL_NAME)
        logger.info("LLM initialized")
        
        # OPTIMIZATION 2: Use cached embeddings
        retriever = load_and_split_documents(file_path)
        
        # OPTIMIZATION 3: Efficient document retrieval
        # Get relevant documents in a single batch
        retrieval_start = time.time()
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        logger.info(f"Retrieved relevant context in {time.time() - retrieval_start:.2f}s")
        
        # OPTIMIZATION 4: Single chain execution
        # Combine template formatting and LLM call into one operation
        chain_start = time.time()
        chain = TEMPLATES["qa"] | llm | StrOutputParser()
        
        # Get chat history once
        chat_history = memory.load_memory_variables({}).get("chat_history", "")
        
        # Single API call for response
        response = chain.invoke({
            "chat_history": chat_history,
            "question": question,
            "context": context
        })
        logger.info(f"Generated response in {time.time() - chain_start:.2f}s")
        
        # Update memory
        memory.save_context(
            {"question": question},
            {"answer": response}
        )
        
        logger.info(f"Total workflow completed in {time.time() - start_time:.2f}s")
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
