from flask import Flask, render_template, request, jsonify, Response
import os
import tempfile
import fitz
from pathlib import Path
import base64
from PIL import Image
import io
import sys
import time
import logging
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path for imports
app_dir = Path(__file__).parent
root_dir = app_dir.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

try:
    from app.memory_manager import MemoryManager
    from app.rag import process_file, load_and_split_documents
    from app.config import MODEL_NAME
    from app.templates import TEMPLATES
except ModuleNotFoundError:
    from memory_manager import MemoryManager
    from rag import process_file, load_and_split_documents
    from config import MODEL_NAME
    from templates import TEMPLATES

app = Flask(__name__)
memory_manager = MemoryManager()

# Configure upload folder
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'study_assistant_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

def display_pdf(file_path):
    """Convert PDF pages to base64 images"""
    start_time = time.time()
    logger.info("Starting PDF conversion...")
    
    doc = fitz.open(file_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        img_str = base64.b64encode(img_byte_arr).decode()
        images.append(img_str)
        logger.info(f"Processed page {page_num + 1}/{len(doc)} in {time.time() - start_time:.2f}s")
    
    doc.close()
    logger.info(f"PDF conversion completed in {time.time() - start_time:.2f}s")
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    logger.info("Starting file upload process...")
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400
    
    if file and file.filename.endswith('.pdf'):
        try:
            # Save uploaded file temporarily
            save_start = time.time()
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            logger.info(f"File saved in {time.time() - save_start:.2f}s")
            
            # Convert PDF to images
            convert_start = time.time()
            images = display_pdf(file_path)
            logger.info(f"PDF conversion completed in {time.time() - convert_start:.2f}s")
            
            # Process the file for embeddings
            process_start = time.time()
            process_file(file_path)
            logger.info(f"File processing completed in {time.time() - process_start:.2f}s")
            
            total_time = time.time() - start_time
            logger.info(f"Total upload process completed in {total_time:.2f}s")
            
            return jsonify({
                'status': 'complete',
                'message': 'File uploaded and processed successfully',
                'images': images,
                'file_path': file_path,
                'processing_time': f"{total_time:.2f}s"
            })
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({'error': str(e), 'status': 'error'}), 500
    
    return jsonify({'error': 'Invalid file type', 'status': 'error'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    logger.info("Starting chat process...")
    
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': 'Invalid request', 'status': 'error'}), 400

    message = data['message']
    files = data.get('files', [])  # List of uploaded filenames

    try:
        # Initialize workflow based on whether we have files
        workflow_name = "Document Q&A" if files else "General Chat"
        logger.info(f"Using workflow: {workflow_name}")

        if files:
            # Multi-document chat
            all_docs = []
            for filename in files:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(file_path):
                    retriever = load_and_split_documents(file_path)
                    docs = retriever.get_relevant_documents(message)
                    all_docs.extend(docs)
                else:
                    logger.warning(f"File not found: {filename}")

            if all_docs:
                # Combine context from all documents
                context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
                
                # Create and execute chain
                llm = ChatOpenAI(model=MODEL_NAME)
                chain = TEMPLATES["qa"] | llm | StrOutputParser()
                response = chain.invoke({
                    "question": message,
                    "context": context,
                    "chat_history": memory_manager.load_memory_variables({}).get("chat_history", "")
                })
            else:
                response = "No relevant documents found to answer your question."
        else:
            # General chat without document context
            llm = ChatOpenAI(model=MODEL_NAME)
            response = llm.invoke([HumanMessage(content=message)]).content

        # Update memory
        memory_manager.save_context(
            {"question": message},
            {"answer": response}
        )

        total_time = time.time() - start_time
        logger.info(f"Chat completed in {total_time:.2f}s")
        
        return jsonify({
            'status': 'complete',
            'answer': response,
            'workflow': workflow_name,
            'processing_time': f"{total_time:.2f}s"
        })

    except Exception as e:
        logger.error(f"Error in chat process: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'workflow': workflow_name
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
