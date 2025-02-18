"""Tests for the RAG pipeline."""
import pytest
from app.rag import RAGPipeline
from app.utils import validate_file_type

def test_validate_file_type():
    """Test file type validation."""
    assert validate_file_type("test.pdf") == True
    assert validate_file_type("test.txt") == True
    assert validate_file_type("test.docx") == True
    assert validate_file_type("test.md") == True
    #assert validate_file_type("test.invalid") == False

def test_rag_pipeline_initialization():
    """Test RAG pipeline initialization."""
    pipeline = RAGPipeline()
    assert pipeline is not None
    assert pipeline.vector_store is None
    assert pipeline.qa_chain is None


