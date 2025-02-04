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
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from app.rag import process_file, load_and_split_documents
from app.config import UPLOAD_FOLDER

def create_test_pdf():
    """Create a test PDF document"""
    test_content = """
    Introduction to Machine Learning
    
    Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data and improve their accuracy over time without being explicitly programmed to do so.
    
    Types of Machine Learning
    
    1. Supervised Learning:
    In supervised learning, the algorithm learns from labeled training data. Each example in the training dataset includes both input features and the desired output. The algorithm learns to map inputs to outputs.
    
    2. Unsupervised Learning:
    Unsupervised learning algorithms work with unlabeled data. They try to find patterns and structures within the data without any predefined output to learn from. Common applications include clustering and dimensionality reduction.
    
    3. Reinforcement Learning:
    This type of learning involves an agent that learns to make decisions by interacting with an environment. The agent receives rewards or penalties for its actions and learns to maximize the cumulative reward.
    
    Common Applications
    
    Machine learning has numerous real-world applications:
    - Image and speech recognition
    - Natural language processing
    - Recommendation systems
    - Autonomous vehicles
    - Medical diagnosis
    - Financial forecasting
    
    Best Practices
    
    When implementing machine learning solutions:
    1. Start with clean, well-prepared data
    2. Choose appropriate algorithms for your problem
    3. Validate your models thoroughly
    4. Monitor model performance in production
    5. Regularly update and retrain models
    
    Conclusion
    
    Machine learning continues to evolve and find new applications across industries. Understanding its fundamentals is crucial for modern software development and data science.
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
