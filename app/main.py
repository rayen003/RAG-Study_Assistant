import streamlit as st
import os
import sys
from pathlib import Path

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import fitz
import tempfile
from PIL import Image
import io
import time
import logging

# Now import our local modules
try:
    from app.memory_manager import MemoryManager
    from app.rag import process_file, load_vector_store
    from app.config import MODEL_NAME
    from app.templates import TEMPLATES
except ModuleNotFoundError:
    from memory_manager import MemoryManager
    from rag import process_file, load_vector_store
    from config import MODEL_NAME
    from templates import TEMPLATES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure upload folder
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'study_assistant_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Initialize session state
if 'memory_manager' not in st.session_state:
    st.session_state.memory_manager = MemoryManager()
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = set()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def display_pdf(file_path):
    """Convert PDF pages to images for display"""
    doc = fitz.open(file_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def main():
    st.set_page_config(
        page_title="Study Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme and better styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #1a1a1a;
            color: white;
        }
        .uploadedFile {
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .chat-message {
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .user-message {
            background-color: #2d2d2d;
        }
        .assistant-message {
            background-color: #4CAF50;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for document upload and management
    with st.sidebar:
        st.title("Study Assistant")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
        
        if uploaded_file:
            file_path = Path(UPLOAD_FOLDER) / uploaded_file.name
            
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("Processing document..."):
                    # Save the file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Process the file
                    if process_file(str(file_path)):
                        st.session_state.uploaded_files.add(uploaded_file.name)
                        st.success("Document processed successfully!")
                    else:
                        st.error("Failed to process document")
        
        # Display uploaded documents
        if st.session_state.uploaded_files:
            st.subheader("Uploaded Documents")
            for filename in st.session_state.uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(filename)
                with col2:
                    if st.button("Preview", key=f"preview_{filename}"):
                        file_path = Path(UPLOAD_FOLDER) / filename
                        if file_path.exists():
                            images = display_pdf(str(file_path))
                            for img in images:
                                st.image(img, use_column_width=True)
        else:
            st.info("No documents uploaded yet")

    # Main chat interface
    st.header("Chat")
    
    # Initialize chat history if not exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Display assistant's "thinking" message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get response from memory manager
                response = st.session_state.memory_manager.chat(
                    user_input=prompt,
                    file_path=str(Path(UPLOAD_FOLDER) / next(iter(st.session_state.uploaded_files))) if st.session_state.uploaded_files else None
                )
                
                # Display the response
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
