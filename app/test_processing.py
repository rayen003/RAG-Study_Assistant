import sys
import time
from pathlib import Path
import logging
from typing import List, Dict
import fitz  # PyMuPDF
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
current_dir = Path(__file__).parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from rag import process_file, load_and_split_documents
from config import UPLOAD_FOLDER

def create_test_pdf():
    """Create a test PDF document"""
    test_content = """
    This is a test document for our Study Assistant.
    It contains multiple paragraphs to test document splitting and processing.
    
    Here's a second paragraph with some technical content.
    We can use this to test our text extraction and chunking.
    
    And a final paragraph to ensure we have enough content to test with.
    This should give us multiple chunks to work with.
    """
    
    pdf_path = os.path.join(UPLOAD_FOLDER, "test_document.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), test_content)
    doc.save(pdf_path)
    doc.close()
    return pdf_path

def test_document_processing():
    """Test the entire document processing pipeline with timing"""
    
    # Create test directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Create a test PDF
    logger.info("Creating test PDF document...")
    test_file = create_test_pdf()
    logger.info(f"Test file created at: {test_file}")
    
    # Test process_file
    logger.info("\n=== Testing Document Processing ===")
    start_time = time.time()
    try:
        process_file(test_file)
        process_time = time.time() - start_time
        logger.info(f"✓ Document processing completed in {process_time:.2f} seconds")
    except Exception as e:
        logger.error(f"✗ Error in process_file: {str(e)}")
        return
    
    # Test load_and_split_documents
    logger.info("\n=== Testing Document Loading and Splitting ===")
    start_time = time.time()
    try:
        retriever = load_and_split_documents(test_file)
        load_time = time.time() - start_time
        logger.info(f"✓ Document loading completed in {load_time:.2f} seconds")
        
        # Test retrieval
        logger.info("\n=== Testing Document Retrieval ===")
        test_queries = [
            "What is this document about?",
            "technical content",
            "final paragraph"
        ]
        
        for query in test_queries:
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
            retrieval_time = time.time() - start_time
            
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Retrieved {len(docs)} chunks in {retrieval_time:.2f} seconds")
            
            for i, doc in enumerate(docs[:2], 1):
                logger.info(f"\nChunk {i}:")
                logger.info(f"Content: {doc.page_content[:200]}...")
                logger.info(f"Source: {doc.metadata.get('source', 'N/A')}")
                logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
    
    except Exception as e:
        logger.error(f"✗ Error in document loading/retrieval: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.remove(test_file)
            logger.info("\n=== Cleanup ===")
            logger.info("✓ Test file removed")
        except Exception as e:
            logger.error(f"✗ Error during cleanup: {str(e)}")

if __name__ == "__main__":
    test_document_processing()
