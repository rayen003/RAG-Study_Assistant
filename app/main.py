import streamlit as st
import os
import tempfile
import sys
from pathlib import Path
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import logging

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Try both import styles
try:
    from app.memory_manager import MemoryManager
    from app.rag import process_file, detect_multimodal_query, execute_workflow
    from app.components.progress import show_progress_message
except ModuleNotFoundError:
    from memory_manager import MemoryManager
    from rag import process_file, detect_multimodal_query, execute_workflow
    from components.progress import show_progress_message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []

class StreamHandler:
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.last_displayed_text = ""
        show_progress_message("Thinking", thinking=True)
        
    def on_llm_start(self, *args, **kwargs):
        self.container.empty()
        self.text = ""
        show_progress_message("Retrieving context", thinking=True)
        
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.text += token
        # Update the display every few tokens to show progress
        if len(self.text) - len(self.last_displayed_text) > 10:
            self.container.markdown(self.text + "")
            self.last_displayed_text = self.text
        
    def on_llm_end(self, *args, **kwargs):
        self.container.markdown(self.text)
        show_progress_message("Response complete")

def get_pdf_pages(file_path: str) -> list:
    """Get all pages of the PDF as base64 encoded images."""
    try:
        pages = []
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Increase resolution
                img_data = pix.tobytes("png")
                encoded = base64.b64encode(img_data).decode()
                pages.append(f"data:image/png;base64,{encoded}")
        return pages
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return []

def display_pdf(file_path):
    """Display a PDF file preview."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="600px"
            style="border: none;">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

def display_message_with_citations(container, message, is_user=False):
    """Display a chat message with hoverable citations."""
    with container.chat_message("human" if is_user else "assistant"):
        if is_user:
            st.markdown(message["content"])
        else:
            text = message.get("content", "")
            citations = {}
            
            # Format citations dictionary
            if "sources" in message:
                for i, source in enumerate(message["sources"], 1):
                    key = f"Source {i}"
                    content = source["text"]
                    if "page" in source:
                        content += f"\n\nPage: {source['page']}"
                    citations[key] = content
            
            if citations:
                # Create columns for the main text and citations
                col1, col2 = st.columns([4, 1])
                with col1:
                    # Handle streaming content
                    if "placeholder" in message:
                        message["placeholder"].markdown(text)
                    else:
                        st.markdown(text)
                with col2:
                    st.markdown("##### Sources")
                    for citation_key, content in citations.items():
                        with st.expander(citation_key):
                            st.markdown(content)
            else:
                # Handle streaming content
                if "placeholder" in message:
                    message["placeholder"].markdown(text)
                else:
                    st.markdown(text)

def show_progress_message(message):
    """Show a progress message to the user."""
    st.markdown(f"""
        <div class='status-message'>
            {message}
        </div>
    """, unsafe_allow_html=True)

def main():
    # Set up the Streamlit app with custom styling
    st.set_page_config(
        page_title="Study Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .chat-message {
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            background-color: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message {
            background-color: #e3f2fd;
        }
        .assistant-message {
            background-color: #f5f5f5;
        }
        .stMarkdown {
            font-size: 16px !important;
            line-height: 1.6 !important;
        }
        .citation {
            font-size: 14px;
            color: #666;
            border-left: 3px solid #1f77b4;
            padding: 10px;
            margin: 10px 0;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .status-message {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            padding: 10px 20px;
            border-radius: 5px;
            background: rgba(0,0,0,0.8);
            color: white;
        }
        .stButton button {
            width: 100%;
        }
        .pdf-preview {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = MemoryManager()
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "show_preview" not in st.session_state:
        st.session_state.show_preview = True
    
    # Create sidebar
    with st.sidebar:
        st.title("📚 Study Assistant")
        st.markdown("---")
        
        # File uploader in sidebar
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf'],
            help="Upload your PDF document"
        )
        
        if uploaded_file and (st.session_state.current_file is None or 
                            uploaded_file.name != os.path.basename(st.session_state.current_file)):
            # Show processing message
            show_progress_message("Processing document...")
            
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            # Process file
            try:
                process_file(file_path)
                st.session_state.current_file = file_path
                show_progress_message("Document processed successfully!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
                st.session_state.current_file = None
        
        # Document preview toggle
        if st.session_state.current_file:
            show_preview = st.toggle("Show Document Preview", value=st.session_state.show_preview)
            st.session_state.show_preview = show_preview
            
            if show_preview:
                with st.container():
                    st.markdown("### Document Preview")
                    st.markdown('<div class="pdf-preview">', unsafe_allow_html=True)
                    display_pdf(st.session_state.current_file)
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat area
    chat_container = st.container()
    
    # Display messages
    for message in st.session_state.messages:
        display_message_with_citations(chat_container, message, message.get("role") == "user")
    
    # User input
    if question := st.chat_input("Ask a question about your document...", 
                               disabled=not st.session_state.current_file):
        # Add user message
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        display_message_with_citations(chat_container, user_message, is_user=True)
        
        if st.session_state.current_file:
            # Create message containers
            thinking_placeholder = chat_container.empty()
            with thinking_placeholder.chat_message("assistant"):
                st.markdown("Thinking...")
            
            # Get streaming response from workflow
            response = execute_workflow(
                st.session_state.memory,
                question,
                st.session_state.current_file
            )
            
            if "answer_generator" in response:
                # Initialize response
                full_response = ""
                
                # Create new assistant message container
                assistant_container = chat_container.chat_message("assistant")
                
                # Create columns for text and citations
                if response.get("sources"):
                    cols = assistant_container.columns([4, 1])
                    text_container = cols[0].empty()
                    
                    # Set up citations
                    with cols[1]:
                        st.markdown("##### Sources")
                        for i, source in enumerate(response["sources"], 1):
                            with st.expander(f"Source {i}"):
                                st.markdown(source["text"])
                                if "page" in source:
                                    st.markdown(f"\n\nPage: {source['page']}")
                else:
                    text_container = assistant_container.empty()
                
                # Remove thinking message
                thinking_placeholder.empty()
                
                # Stream the response
                for chunk in response["answer_generator"]:
                    if hasattr(chunk, "content"):
                        content = chunk.content
                    else:
                        content = str(chunk)
                    full_response += content
                    text_container.markdown(full_response)
                
                # Save final message
                final_message = {
                    "role": "assistant",
                    "content": full_response,
                    "sources": response.get("sources", [])
                }
                st.session_state.messages.append(final_message)
                
                # Update memory
                st.session_state.memory.save_context(
                    {"question": question},
                    {"answer": full_response}
                )
            else:
                # Handle non-streaming response
                thinking_placeholder.empty()
                assistant_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", [])
                }
                st.session_state.messages.append(assistant_message)
                display_message_with_citations(chat_container, assistant_message)

if __name__ == "__main__":
    main()
