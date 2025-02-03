from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List, Optional, Union, Dict
from langchain_core.output_parsers.string import StrOutputParser
from pydantic import BaseModel
import importlib
import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try both import styles
try:
    from app.config import MODEL_NAME
    from app.templates import TEMPLATES
    from app.memory_manager import MemoryManager
except ModuleNotFoundError:
    from config import MODEL_NAME
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

def process_file(file_content: bytes, filename: str) -> Optional[FileInput]:
    """Process file content and return FileInput if valid."""
    extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    file_type = None
    for type_name, extensions in FILE_TYPE_MAP.items():
        if extension in extensions:
            file_type = type_name
            break
    
    if not file_type:
        return None
        
    return FileInput(
        content=file_content,
        file_type=file_type,
        filename=filename
    )

def detect_multimodal_query(query: str) -> bool:
    """Detect if a query requires multimodal processing."""
    multimodal_keywords = [
        'image', 'picture', 'photo', 'document', 'file',
        'look at', 'analyze', 'show', 'read', 'scan'
    ]
    return any(keyword in query.lower() for keyword in multimodal_keywords)

def execute_workflow(memory, question: str, file_input: Optional[FileInput] = None):
    """Execute the appropriate workflow based on input type."""
    try:
        # Determine workflow based on input
        workflow_name = "Multimodal" if file_input or detect_multimodal_query(question) else "General"
        
        # Import the appropriate workflow module
        module_path = WORKFLOW_MAP[workflow_name]
        try:
            # Try absolute import first
            module = importlib.import_module(f"app.workflows{module_path}")
        except ModuleNotFoundError:
            # Fall back to relative import
            module = importlib.import_module(f"workflows{module_path}", package="app")
        
        # Get the workflow function
        workflow_func = getattr(module, module_path.split('.')[-1])
        
        # Execute the workflow
        return workflow_func(memory, question, file_input)
        
    except Exception as e:
        return {
            "answer": f"Error executing workflow: {str(e)}",
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
                        file_input = process_file(file_content, file_path)
                        if not file_input:
                            print("Warning: Unsupported file type. Proceeding without file.")
                
                # Execute workflow and get response
                response = execute_workflow(memory, question, file_input)
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
