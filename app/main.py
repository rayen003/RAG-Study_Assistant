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
    from app.chat_manager import ChatManager
    from app.rag import process_file, load_vector_store
    from app.config import MODEL_NAME, UPLOAD_FOLDER
    from app.templates import TEMPLATES
except ModuleNotFoundError:
    from chat_manager import ChatManager
    from rag import process_file, load_vector_store
    from config import MODEL_NAME, UPLOAD_FOLDER
    from templates import TEMPLATES

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to the upload folder."""
    try:
        # Ensure upload folder exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Generate a unique filename
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File saved: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        st.error(f"Error saving file: {e}")
        return None

def display_pdf(file_path):
    """Convert PDF pages to images for display"""
    pdf_document = fitz.open(file_path)
    images = []
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        # Increase zoom factor for larger preview (2 means 200% of original size)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Convert to bytes for Streamlit
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        images.append(img_bytes)
    
    return images

def main():
    st.set_page_config(layout="wide", page_title="Study Assistant", initial_sidebar_state="collapsed")

    # Custom CSS for clean layout
    st.markdown("""
        <style>
        .main > div {
            padding: 0 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            width: 100%;
        }
        .user-message {
            background-color: #2D2D2D;
        }
        .assistant-message {
            background-color: #0E4429;
        }
        .thinking-message {
            background-color: #1E1E1E;
            color: #888;
            font-style: italic;
        }
        .document-section {
            background-color: #1E1E1E;
            border-radius: 0.5rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .chat-container {
            background-color: #1E1E1E;
            border-radius: 0.5rem;
            padding: 1rem;
            height: calc(100vh - 120px);
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }
        .chat-messages {
            flex-grow: 1;
            overflow-y: auto;
            margin-bottom: 1rem;
        }
        .chat-input {
            display: flex;
            gap: 0.5rem;
            padding: 0.5rem;
            background: #2D2D2D;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager(MODEL_NAME)
    if 'thinking' not in st.session_state:
        st.session_state.thinking = False

    # Create two columns: left for document, right for chat
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("Document Preview")
        
        # File uploader in the document section
        uploaded_file = st.file_uploader(
            "Upload a PDF file",
            type=["pdf"],
            key="file_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                with st.spinner("Processing document..."):
                    # Process document and store the vector store path
                    vector_store_path = process_file(file_path)
                    if vector_store_path:
                        # Load the vector store
                        vector_store = load_vector_store(vector_store_path)
                        if vector_store:
                            st.session_state.vector_store = vector_store
                            st.success("Document processed successfully!")
                            
                            # Display PDF preview
                            images = display_pdf(file_path)
                            for img in images:
                                st.image(img, use_column_width=True)
                        else:
                            st.error("Failed to load the document store")
                    else:
                        st.error("Failed to process the document")

    with right_col:
        st.subheader("Chat")
        
        # Chat messages container
        chat_placeholder = st.container()
        
        # Display messages
        with chat_placeholder:
            for message in st.session_state.messages:
                message_class = "user-message" if message["role"] == "user" else "assistant-message"
                st.markdown(f"""
                    <div class="chat-message {message_class}">
                        {message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            
            # Show thinking message if processing
            if st.session_state.get('processing', False):
                st.markdown("""
                    <div class="chat-message thinking-message">
                        Thinking...
                    </div>
                """, unsafe_allow_html=True)

        # Chat input
        with st.container():
            cols = st.columns([6, 1])
            with cols[0]:
                # Initialize the key in session state if it doesn't exist
                if "user_input_key" not in st.session_state:
                    st.session_state.user_input_key = 0
                
                user_input = st.text_input(
                    "Your message",
                    key=f"user_input_{st.session_state.user_input_key}",
                    label_visibility="collapsed"
                )
            with cols[1]:
                send_button = st.button("Send", use_container_width=True)

        if send_button and user_input:
            if not st.session_state.get('processing', False):
                st.session_state.processing = True
                
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                try:
                    # Get context from vector store if available
                    context = ""
                    if hasattr(st.session_state, 'vector_store') and st.session_state.vector_store:
                        # Get relevant chunks from vector store
                        results = st.session_state.vector_store.similarity_search(user_input, k=3)
                        context = "\n".join([doc.page_content for doc in results])
                    
                    # Get AI response with context
                    response = st.session_state.chat_manager.get_response(user_input, context)
                    # Add AI response
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                
                # Clear input by incrementing the key
                st.session_state.user_input_key += 1
                # Clear processing state
                st.session_state.processing = False
                st.rerun()

if __name__ == "__main__":
    main()
