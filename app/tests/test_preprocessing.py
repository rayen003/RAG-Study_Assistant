import sys
import time
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent.parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from app.rag import process_file, load_and_split_documents

def test_document_processing(file_path):
    """Test document processing pipeline and measure performance"""
    
    logger.info(f"Testing document processing for: {file_path}")
    
    # Test process_file
    start_time = time.time()
    try:
        process_file(file_path)
        process_time = time.time() - start_time
        logger.info(f"Document processing completed in {process_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in process_file: {str(e)}")
        return
    
    # Test load_and_split_documents
    start_time = time.time()
    try:
        retriever = load_and_split_documents(file_path)
        load_time = time.time() - start_time
        logger.info(f"Document loading and splitting completed in {load_time:.2f} seconds")
        
        # Test retrieval
        test_query = "What is this document about?"
        start_time = time.time()
        docs = retriever.get_relevant_documents(test_query)
        retrieval_time = time.time() - start_time
        
        logger.info(f"Retrieved {len(docs)} relevant chunks in {retrieval_time:.2f} seconds")
        logger.info("\nSample chunks:")
        for i, doc in enumerate(docs[:2], 1):
            logger.info(f"\nChunk {i}:")
            logger.info(f"Content: {doc.page_content[:200]}...")
            logger.info(f"Source: {doc.metadata.get('source', 'N/A')}")
            logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
    
    except Exception as e:
        logger.error(f"Error in document loading/retrieval: {str(e)}")

if __name__ == "__main__":
    # Test with a sample PDF
    test_file = "/path/to/your/test.pdf"  # Replace with actual test file path
    test_document_processing(test_file)
