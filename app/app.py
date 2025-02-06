import streamlit as st
from pathlib import Path
import logging
import os
import shutil
import base64
import tempfile
import uuid

from components.progress import show_progress_message
from embeddings import LocalEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF preview

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = []
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        show_progress_message("Thinking", thinking=True)
        
    def on_llm_start(self, **kwargs):
        show_progress_message("Retrieving context", thinking=True)
        
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

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

def cleanup_previous_document():
    """Clean up resources from the previous document."""
    # Clear conversation and chat history
    st.session_state.conversation = None
    st.session_state.messages = []
    
    # Clear PDF pages
    st.session_state.pdf_pages = []
    
    # Clear vector database
    st.session_state.vectordb = None

def initialize_vectordb(uploaded_file):
    """Initialize the vector database with the uploaded document."""
    # Clean up previous document
    cleanup_previous_document()
    st.session_state.current_doc = uploaded_file.name
    
    show_progress_message("Processing document")
    
    # Use a temporary directory for file handling
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file temporarily
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Log file details for debugging
        file_size = os.path.getsize(temp_path)
        logger.info(f"Uploaded file: {uploaded_file.name}")
        logger.info(f"File size: {file_size} bytes")
        
        # Get PDF preview
        if uploaded_file.name.endswith('.pdf'):
            st.session_state.pdf_pages = get_pdf_pages(temp_path)
        
        # Load document based on file type
        show_progress_message("Loading document")
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            
            documents = loader.load()
            
            # Log document details
            logger.info(f"Number of documents loaded: {len(documents)}")
            if documents:
                logger.info(f"First document preview: {documents[0].page_content[:500]}")
        except Exception as e:
            logger.error(f"Error loading document: {e}", exc_info=True)
            st.error(f"Could not load document: {e}")
            return None
        
        # Split documents
        show_progress_message("Analyzing content")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Log splits details
        logger.info(f"Number of document splits: {len(splits)}")
        if splits:
            logger.info(f"First split preview: {splits[0].page_content[:500]}")
        
        # Create embeddings and store in vectordb
        show_progress_message("Creating embeddings")
        embeddings = LocalEmbeddings()
        
        # Create FAISS vector store
        try:
            vectordb = FAISS.from_documents(splits, embeddings)
            
            # Store in session state
            st.session_state.vectordb = vectordb
            
            return vectordb
        except Exception as e:
            logger.error(f"Error creating vector store: {e}", exc_info=True)
            st.error(f"Could not create vector store: {e}")
            return None

def initialize_conversation(vectordb):
    """Initialize the conversation chain."""
    show_progress_message("Initializing")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please check your .env file.")
        st.stop()
    
    # Initialize memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )
    
    # Initialize conversation chain with more verbose settings
    conversation = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            temperature=0.7,  # Slightly more creative
            max_tokens=1000,  # Even longer responses
            api_key=os.getenv("OPENAI_API_KEY"),
            streaming=True
        ),
        retriever=vectordb.as_retriever(
            search_kwargs={
                "k": 6  # Increase number of retrieved documents
            }
        ),
        memory=memory,
        return_source_documents=True,
        return_generated_question=True,
        verbose=True  # For debugging
    )
    
    return conversation

def format_response_with_citations(response_text: str, source_docs: list) -> tuple[str, dict]:
    """Format the response with numbered citations and create a citations map."""
    citations = {}
    formatted_text = response_text
    
    # Create citations map
    for i, doc in enumerate(source_docs, 1):
        citation_key = f"[{i}]"
        citations[citation_key] = doc.page_content[:200] + "..."  # Truncate long content
        
        # Add citation numbers to the text
        if doc.page_content[:50] in formatted_text:  # Simple check for content presence
            formatted_text = formatted_text.replace(doc.page_content[:50], f" {citation_key} ")
    
    return formatted_text, citations

def display_message_with_citations(container, text: str, citations: dict = None, is_user: bool = False):
    """Display a chat message with hoverable citations."""
    with container.chat_message("human" if is_user else "assistant"):
        if citations:
            # Create columns for the main text and citations
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(text)
            with col2:
                st.markdown("##### Sources")
                for citation_key, content in citations.items():
                    with st.expander(citation_key):
                        st.write(content)
        else:
            st.write(text)

def main():
    # Configure page layout
    st.set_page_config(layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("Study Assistant")
        
        # File upload in sidebar
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["pdf", "txt"],
            help="Upload a PDF or text file to study",
            label_visibility="collapsed"
        )
    
    # Main chat area
    chat_container = st.container()
    
    # Process uploaded file
    if uploaded_file:
        try:
            if "vectordb" not in st.session_state or st.session_state.current_doc != uploaded_file.name:
                vectordb = initialize_vectordb(uploaded_file)
                if vectordb:
                    st.session_state.conversation = initialize_conversation(vectordb)
                    show_progress_message("Ready")
            
            # Show document preview in sidebar
            if st.session_state.pdf_pages:
                with st.sidebar:
                    st.markdown("### Document Preview")
                    for i, page in enumerate(st.session_state.pdf_pages):
                        st.markdown(f"""
                            <div style='display: flex; justify-content: center;'>
                                <img src='{page}' style='max-width: 100%; margin: 10px 0;'>
                            </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing document: {e}")
            logger.error(f"Document processing error: {e}", exc_info=True)
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            display_message_with_citations(
                chat_container,
                message["content"],
                message.get("citations"),
                message["is_user"]
            )
    
    # Chat input
    if st.session_state.conversation and (question := st.chat_input(
        placeholder="Ask a question about your document",
        key="chat_input"
    )):
        # Add user message
        st.session_state.messages.append({
            "content": question,
            "is_user": True
        })
        
        # Create message placeholder for assistant
        with chat_container.chat_message("assistant"):
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)
            
            try:
                # Get response from conversation chain
                response = st.session_state.conversation(
                    {"question": question},
                    callbacks=[stream_handler]
                )
                
                # Format response with citations
                formatted_response, citations = format_response_with_citations(
                    stream_handler.text,
                    response.get('source_documents', [])
                )
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "content": formatted_response,
                    "citations": citations,
                    "is_user": False
                })
                
                # Rewrite the entire chat history to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
                logger.error(f"Question processing error: {e}", exc_info=True)
                message_placeholder.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
