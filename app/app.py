"""
Streamlit application for document Q&A
"""
import streamlit as st
from pathlib import Path
import logging
import os
import shutil
import base64
import tempfile
import uuid
import time

from components.progress import show_progress_message, clear_progress_messages, init_progress_containers
from embeddings import LocalEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, ConversationChain
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
if 'workflow_type' not in st.session_state:
    st.session_state.workflow_type = "Chat"
if 'thinking_container' not in st.session_state:
    st.session_state.thinking_container = None

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        
    def on_llm_start(self, **kwargs):
        self.text = "Thinking..."
        self.container.markdown(self.text)
        
    def on_llm_new_token(self, token: str, **kwargs):
        try:
            self.text = token if self.text == "Thinking..." else self.text + token
            self.container.markdown(self.text)
        except Exception as e:
            logger.error(f"Error in StreamHandler: {e}")
            
    def on_llm_end(self, **kwargs):
        pass
        
    def on_llm_error(self, error: Exception, **kwargs):
        logger.error(f"LLM Error in StreamHandler: {error}")

def get_pdf_pages(file_path: str):
    """Get all pages of the PDF as base64 encoded images."""
    images = []
    try:
        pdf_document = fitz.open(file_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode()
            images.append(f"data:image/png;base64,{img_base64}")
        pdf_document.close()
    except Exception as e:
        logger.error(f"Error processing PDF pages: {e}")
    return images

def cleanup_previous_document():
    """Clean up resources from the previous document."""
    if st.session_state.current_doc:
        try:
            # Clear PDF pages
            st.session_state.pdf_pages = []
            # Clear conversation history
            st.session_state.messages = []
            # Clear vector store
            st.session_state.vectordb = None
            # Clear conversation chain
            st.session_state.conversation = None
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def initialize_vectordb(uploaded_file):
    """Initialize the vector database with the uploaded document."""
    try:
        # Create a temporary directory for document processing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save uploaded file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Set workflow type based on file type
            if file_path.lower().endswith('.pdf'):
                st.session_state.workflow_type = "Multimodal"
                loader = PyPDFLoader(file_path)
                # Store PDF pages for preview
                st.session_state.pdf_pages = get_pdf_pages(file_path)
            else:
                st.session_state.workflow_type = "Text"
                loader = TextLoader(file_path)
            
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = LocalEmbeddings()
            vectordb = FAISS.from_documents(splits, embeddings)
            
            # Update session state
            st.session_state.current_doc = uploaded_file.name
            return vectordb
            
    except Exception as e:
        st.error(f"Could not create vector store: {str(e)}")
        logger.error(f"Vector store creation error: {e}", exc_info=True)
        return None

def initialize_conversation(vectordb=None):
    """Initialize the conversation chain."""
    try:
        # Create LLM
        llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4-1106-preview"
        )
        
        if vectordb:
            # Create memory for retrieval chain
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True
            )
            
            # Create chain with retrieval
            chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectordb.as_retriever(),
                memory=memory,
                return_source_documents=True,
                chain_type="stuff",
                combine_docs_chain_kwargs={"output_key": "answer"}
            )
            st.session_state.workflow_type = "Multimodal" if st.session_state.get('current_doc', '').lower().endswith('.pdf') else "Text"
        else:
            # Create memory for basic chain
            memory = ConversationBufferMemory(
                input_key="input",
                memory_key="history"
            )
            
            # Create simple conversation chain without retrieval
            chain = ConversationChain(
                llm=llm,
                memory=memory,
                verbose=True
            )
            st.session_state.workflow_type = "Chat"
        
        return chain
        
    except Exception as e:
        st.error(f"Could not initialize conversation: {str(e)}")
        logger.error(f"Conversation initialization error: {e}", exc_info=True)
        return None

def format_response_with_citations(response_text: str, source_docs: list):
    """Format the response with numbered citations and create a citations map."""
    formatted_text = response_text
    citations = {}
    
    for i, doc in enumerate(source_docs):
        citation_key = f"[{i + 1}]"
        citations[citation_key] = doc.page_content
        
        # Add citation numbers to the text
        if doc.page_content[:50] in formatted_text:  # Simple check for content presence
            formatted_text = formatted_text.replace(doc.page_content[:50], f" {citation_key} ")
    
    return formatted_text, citations

def display_message_with_citations(container, text: str, citations: dict = None, is_user: bool = False):
    """Display a chat message with hoverable citations."""
    message_container = container.chat_message("human" if is_user else "assistant")
    if citations:
        # Create columns for the main text and citations
        col1, col2 = message_container.columns([4, 1])
        with col1:
            st.markdown(text)
        with col2:
            st.markdown("##### Sources")
            for citation_key, content in citations.items():
                with st.expander(citation_key):
                    st.markdown(content)
    else:
        message_container.markdown(text)
    return message_container

def main():
    # Configure page layout
    st.set_page_config(layout="wide")
    
    # Initialize progress containers
    init_progress_containers()
    
    # Initialize session state
    if 'workflow_type' not in st.session_state:
        st.session_state.workflow_type = "Chat"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation' not in st.session_state:
        from app.workflows.general import GeneralWorkflow
        st.session_state.conversation = GeneralWorkflow()
    
    st.markdown("""
        <style>
        .workflow-badge {
            background-color: #0e1117;
            padding: 0.2rem 0.6rem;
            border-radius: 0.5rem;
            border: 1px solid #31333f;
            font-size: 0.8rem;
            float: right;
            margin-top: 0.5rem;
        }
        .sources-header {
            font-size: 0.7rem !important;
            opacity: 0.8;
            margin-bottom: 0.2rem !important;
            text-align: right;
        }
        .sources-header.compact {
            font-size: 0.6rem !important;
            opacity: 0.7;
            margin-bottom: 0.1rem !important;
        }
        .stExpander {
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
            padding: 0 !important;
            margin: 0 !important;
        }
        .stExpander > div[role="button"] {
            padding: 0.1rem !important;
            color: #666 !important;
            background: none !important;
            border: none !important;
            font-size: 0.6rem !important;
        }
        .stExpander > div[data-testid="stExpander"] {
            padding: 0.1rem !important;
            font-size: 0.6rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.title("Study Assistant")
    with header_col2:
        st.markdown(f"""
            <div class="workflow-badge">
                Current Workflow: {st.session_state.workflow_type}
            </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Upload Document")
        st.markdown("*Optional: Upload a document to enable document-based question answering*")
        
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
            if st.session_state.current_doc != uploaded_file.name:
                phase_container = st.empty()
                
                # Phase 1: Processing document
                with phase_container:
                    st.info("📄 Processing document...")
                vectordb = initialize_vectordb(uploaded_file)
                
                if vectordb:
                    # Phase 2: Creating embeddings
                    with phase_container:
                        st.info("🔄 Creating embeddings...")
                    st.session_state.conversation = initialize_conversation(vectordb)
                    
                    # Phase 3: Ready
                    with phase_container:
                        st.success("✅ Ready!")
                    time.sleep(1)  # Show ready message briefly
                    phase_container.empty()  # Clear the phase message
        
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
    
    # Initialize conversation if not already done
    if st.session_state.conversation is None:
        st.session_state.conversation = initialize_conversation()
    
    # Chat input
    if (question := st.chat_input(
        placeholder="Ask a question..." if not uploaded_file else "Ask a question about your document",
        key="chat_input"
    )):
        # Add user message immediately
        st.session_state.messages.append({
            "content": question,
            "is_user": True
        })
        
        # Display chat history and handle response
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                if message["is_user"]:
                    st.chat_message("user").markdown(message["content"])
                else:
                    msg = st.chat_message("assistant")
                    msg.markdown(message["content"])
                    if message.get("citations"):
                        cols = msg.columns([4, 1])
                        with cols[1]:
                            st.markdown("**Sources**")
                            for i, (key, content) in enumerate(message["citations"].items()):
                                with st.expander(f"[{i+1}] {key}", expanded=False):
                                    st.markdown(content)
            
            # Create a persistent container for the current message
            response_container = st.container()
            
            # Handle new message
            if question:  
                print("\n=== Starting new response generation ===")
                
                # Show thinking message in a placeholder
                with response_container:
                    thinking_placeholder = st.empty()
                    with thinking_placeholder:
                        st.chat_message("assistant").markdown("Thinking...")
                
                try:
                    # Prepare input based on workflow type
                    if st.session_state.workflow_type == "Chat":
                        chain_input = {"input": question}
                    else:
                        chain_input = {"question": question}
                    
                    print(f"Chain input: {chain_input}")
                    
                    # Get response
                    print("Getting response from conversation...")
                    response = st.session_state.conversation(chain_input)
                    print(f"Raw response: {response}")
                    
                    # Get answer based on workflow type
                    if st.session_state.workflow_type == "Chat":
                        answer = response["response"]
                        citations = None
                    else:
                        answer = response["answer"]
                        citations = format_response_with_citations(
                            answer,
                            response.get("source_documents", [])
                        )[1]
                    
                    print(f"Processed answer: {answer}")
                    
                    # Replace thinking message with response
                    thinking_placeholder.empty()
                    with thinking_placeholder:
                        msg = st.chat_message("assistant")
                        if citations:
                            cols = msg.columns([6, 1])
                            cols[0].markdown(answer)
                            with cols[1]:
                                st.markdown('<p class="sources-header compact">Sources</p>', unsafe_allow_html=True)
                                for i, (key, content) in enumerate(citations.items()):
                                    with st.expander(f"[{i+1}] {key}", expanded=False):
                                        st.markdown(content)
                        else:
                            msg.markdown(answer)
                    print("Displayed response")
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "content": answer,
                        "citations": citations,
                        "is_user": False
                    })
                    print("Added to message history")
                    
                except Exception as e:
                    print(f"\n=== Error occurred ===\n{str(e)}")
                    st.error(f"Error generating response: {str(e)}")
                    logger.error(f"Response generation error: {e}", exc_info=True)
                    
                    # Replace thinking message with error
                    thinking_placeholder.empty()
                    with thinking_placeholder:
                        st.chat_message("assistant").markdown("Sorry, I encountered an error while generating the response.")
    else:
        # Display existing chat history
        with chat_container:
            for message in st.session_state.messages:
                if message["is_user"]:
                    st.chat_message("user").markdown(message["content"])
                else:
                    msg = st.chat_message("assistant")
                    msg.markdown(message["content"])
                    if message.get("citations"):
                        cols = msg.columns([4, 1])
                        with cols[1]:
                            st.markdown("**Sources**")
                            for i, (key, content) in enumerate(message["citations"].items()):
                                with st.expander(f"[{i+1}] {key}", expanded=False):
                                    st.markdown(content)

if __name__ == "__main__":
    main()
