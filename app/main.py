import streamlit as st
import os
import tempfile
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try both import styles
try:
    from app.memory_manager import MemoryManager
    from app.rag import process_file, detect_multimodal_query, execute_workflow
except ModuleNotFoundError:
    from memory_manager import MemoryManager
    from rag import process_file, detect_multimodal_query, execute_workflow

# Initialize session state
if "memory" not in st.session_state:
    st.session_state.memory = MemoryManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_workflow" not in st.session_state:
    st.session_state.current_workflow = "General"

def get_file_type_info():
    """Return supported file types information."""
    return {
        'image': ['.png', '.jpg', '.jpeg', '.gif'],
        'document': ['.pdf', '.txt', '.doc', '.docx']
    }

def main():
    # Set up the Streamlit app with custom styling
    st.set_page_config(page_title="Study Assistant", layout="wide")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .workflow-info {
            padding: 10px;
            border-radius: 5px;
            background-color: #f0f2f6;
            margin: 10px 0;
        }
        .file-uploader {
            border: 2px dashed #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title
    st.title("🎓 Intelligent Study Assistant")

    # Sidebar for file upload and workflow info
    with st.sidebar:
        st.header("📁 File Upload")
        st.markdown("### Supported File Types")
        
        # Display supported file types
        file_types = get_file_type_info()
        for category, extensions in file_types.items():
            st.markdown(f"**{category.title()}**: {', '.join(extensions)}")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Drag and drop files here",
            type=[ext[1:] for exts in file_types.values() for ext in exts],
            help="Upload your study materials here"
        )

        # Display current workflow
        st.markdown("### 🔄 Current Workflow")
        st.markdown(f"**Active**: {st.session_state.current_workflow}")

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Process uploaded file
        file_input = None
        if uploaded_file is not None:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Process the file
            with open(tmp_file_path, 'rb') as f:
                file_content = f.read()
                file_input = process_file(file_content, uploaded_file.name)
                if file_input:
                    st.success(f"File processed: {uploaded_file.name}")
                else:
                    st.warning("Unsupported file type")

            # Clean up
            os.unlink(tmp_file_path)

        # Get user input
        user_input = st.chat_input("Ask a question about your studies...")
        
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})

            # Determine workflow
            workflow_type = "Multimodal" if file_input or detect_multimodal_query(user_input) else "General"
            st.session_state.current_workflow = workflow_type

            # Execute workflow and get response
            response = execute_workflow(st.session_state.memory, user_input, file_input)

            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"]
            })

            st.rerun()

    with col2:
        st.header("ℹ️ Assistant Info")
        
        # Display workflow information
        st.markdown("""
        ### Workflow Types
        
        1. **General Workflow**
           - Handles text-based queries
           - Provides detailed explanations
           - Uses conversation history
        
        2. **Multimodal Workflow**
           - Processes images and documents
           - Analyzes visual content
           - Handles file-based queries
        
        The assistant automatically switches between workflows based on:
        - Uploaded files
        - Query content
        - User intentions
        """)

if __name__ == "__main__":
    main()
