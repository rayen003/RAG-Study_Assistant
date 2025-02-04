from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
import json
import base64
import fitz
import tempfile
import PIL
from pathlib import Path
import io
import sys
import time
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from werkzeug.utils import secure_filename

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
    from app.rag import process_file
    from app.config import MODEL_NAME
    from app.templates import TEMPLATES
except ModuleNotFoundError:
    from memory_manager import MemoryManager
    from rag import process_file
    from config import MODEL_NAME
    from templates import TEMPLATES

app = Flask(__name__)
memory_manager = MemoryManager()

# Configure upload folder
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'study_assistant_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

def allowed_file(filename):
    return filename.endswith('.pdf')

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
        img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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

@app.route('/preview/<filename>')
def preview_document(filename):
    """Generate preview images for a PDF document."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Convert PDF pages to base64 images
        images = display_pdf(file_path)
        return jsonify({
            'status': 'success',
            'images': images
        })
        
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload with progress tracking."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Save file first
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save file directly
        file.save(file_path)
        
        # Process file after saving
        success = process_file(file_path)
        
        if success:
            return jsonify({
                'status': 'complete',
                'filename': filename
            })
        else:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'status': 'error',
                'error': 'Failed to process file'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with streaming response."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        message = data['message']
        files = data.get('files', [])
        
        def generate():
            try:
                # Initialize chat
                if files:
                    # Multi-document chat
                    all_docs = []
                    for filename in files:
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        if os.path.exists(file_path):
                            retriever = load_and_split_documents(file_path)
                            docs = retriever.get_relevant_documents(message)
                            all_docs.extend(docs)
                    
                    if all_docs:
                        # Combine context from all documents
                        context = "\n\n---\n\n".join([doc.page_content for doc in all_docs])
                        
                        # Create and execute chain
                        llm = ChatOpenAI(
                            model_name=MODEL_NAME,
                            temperature=0.7,
                            streaming=True
                        )
                        chain = TEMPLATES["qa"] | llm | StrOutputParser()
                        for chunk in chain.stream({
                            "question": message,
                            "context": context,
                            "chat_history": memory_manager.load_memory_variables({}).get("chat_history", "")
                        }):
                            yield f"data: {json.dumps({'token': chunk})}\n\n"
                    else:
                        yield f"data: {json.dumps({'token': 'No relevant documents found to answer your question.'})}\n\n"
                else:
                    # General chat
                    llm = ChatOpenAI(
                        model_name=MODEL_NAME,
                        temperature=0.7,
                        streaming=True
                    )
                    for chunk in llm.stream([HumanMessage(content=message)]):
                        if chunk.content:
                            yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                
                # Signal completion
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return Response(generate(), mimetype='text/event-stream')
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
