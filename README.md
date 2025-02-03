# Study Assistant

A smart study assistant powered by RAG (Retrieval Augmented Generation) that helps students better understand their study materials. This application uses OpenAI's GPT-4 model and embeddings to provide intelligent responses to questions about uploaded PDF documents.

## Features

- PDF document upload and processing
- Intelligent question answering using RAG (Retrieval Augmented Generation)
- Interactive chat interface using Streamlit
- Context-aware responses based on document content
- Modern and user-friendly interface
- Support for multiple file types (PDF, images, text documents)
- Automatic workflow selection based on query type
- Conversation memory for contextual responses

## Prerequisites

- Python 3.8+
- OpenAI API key

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/rayen003/RAG-Powered-Study-Assistant.git
   cd RAG-Powered-Study-Assistant
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## Usage

1. Start the application:
   ```bash
   streamlit run app/main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Use the application:
   - Upload documents or images using the sidebar
   - Ask questions in the chat interface
   - The system will automatically choose between:
     - General workflow for text-based queries
     - Multimodal workflow for file-based queries

## Project Structure

```
study-assistant/
├── app/
│   ├── main.py          # Streamlit app entry point
│   ├── rag.py           # RAG pipeline implementation
│   ├── config.py        # Configuration settings
│   ├── memory_manager.py # Memory management
│   ├── templates.py     # Prompt templates
│   └── workflows/       # Workflow implementations
│       ├── general_workflow.py
│       └── multimodal_workflow.py
├── .env.example         # Example environment variables
└── requirements.txt     # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
