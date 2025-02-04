import streamlit as st
import os
import tempfile
import sys
from pathlib import Path
import base64
import fitz  # PyMuPDF
from PIL import Image
import io

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
if "pdf_display" not in st.session_state:
    st.session_state.pdf_display = None
if "current_question" not in st.session_state:
    st.session_state.current_question = None

def get_file_type_info():
    """Return supported file types information."""
    return {
        'document': ['.pdf', '.txt', '.doc', '.docx']
    }

def display_pdf(file_path):
    """Display PDF pages as images"""
    doc = fitz.open(file_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Increase the resolution for better quality
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Convert to base64
        img_str = base64.b64encode(img_byte_arr).decode()
        images.append(f'<img src="data:image/png;base64,{img_str}" style="width:100%; margin-bottom:10px;">')
    
    doc.close()
    return "".join(images)

def main():
    # Set up the Streamlit app with custom styling
    st.set_page_config(page_title="Study Assistant", layout="wide")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            align-self: flex-end;
        }
        .assistant-message {
            background-color: #f5f5f5;
            align-self: flex-start;
        }
        .pdf-viewer {
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 20px;
            height: calc(100vh - 150px);
            overflow-y: auto;
        }
        .chat-input {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 100%;
            padding: 20px;
            background-color: white;
            border-top: 1px solid #ddd;
            z-index: 1000;
        }
        .chat-container {
            margin-bottom: 100px;  /* Space for fixed chat input */
            overflow-y: auto;
            height: calc(100vh - 200px);
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two columns: left for PDF viewer, right for chat
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📁 Document Viewer")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=['pdf'],
            help="Upload your PDF document"
        )

        # Display PDF viewer
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Create a container for the PDF viewer with scrolling
            st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
            st.markdown(display_pdf(tmp_file_path), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Store file path for processing
            st.session_state.current_file = tmp_file_path

    with col2:
        st.header("💬 Chat Interface")
        
        # Create a container for chat messages
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Create a container for the chat input at the bottom
        with st.container():
            st.markdown('<div class="chat-input">', unsafe_allow_html=True)
            # Chat input
            if question := st.chat_input("Ask a question about your document"):
                # Store the current question
                st.session_state.current_question = question
                
                # Display user message immediately
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(question)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Process the question if we have a file
                if hasattr(st.session_state, 'current_file'):
                    with chat_container:
                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            
                            # Get the response from the multimodal workflow
                            response = execute_workflow(
                                st.session_state.memory,
                                question,
                                st.session_state.current_file
                            )
                            
                            # Update the message in real-time
                            message_placeholder.markdown(response["answer"])
                            
                            # Add to chat history
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response["answer"]}
                            )
                else:
                    st.warning("Please upload a document first.")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
