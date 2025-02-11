# Study Assistant:

A powerful AI-powered study assistant that helps you understand and interact with your documents.

## Features:

- 📚 Multi-document support
- 🔍 Semantic search and retrieval
- 💬 Interactive chat interface
- 📊 Real-time processing status
- 👀 Document preview functionality
- 🎨 Modern dark theme UI

## Performance Metrics:

Document processing pipeline performance:
- Document Loading: ~0.2s
- Text Splitting: <0.1s
- Embedding Generation: ~2.5s
- Query Response: 1.2-2.3s

## Setup:

1. Clone the repository
```bash
git clone https://github.com/yourusername/Study-Assistant-Clean.git
cd Study-Assistant-Clean
```

2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Start the server:
```bash
python app/web_app.py
```

2. Open your browser and navigate to `http://localhost:5001`

3. Upload your documents using the attachment button

4. Start chatting with your documents!

## Project Structure

```
Study-Assistant-Clean/
├── app/
│   ├── templates/
│   │   └── index.html      # Web interface
│   ├── config.py           # Configuration settings
│   ├── memory_manager.py   # Chat history management
│   ├── rag.py             # Document processing and retrieval
│   ├── test_processing.py # Performance testing
│   └── web_app.py         # Flask server
├── uploads/               # Document storage
├── .env                  # Environment variables
└── requirements.txt      # Python dependencies
```

## Testing:

Run the test suite to verify document processing:
```bash
python app/test_processing.py
```

This will test:
- Document loading and splitting
- Embedding generation
- Query retrieval
- Processing performance

## Recent Updates:

- Added multi-document support
- Implemented document preview functionality
- Added real-time processing status indicators
- Updated UI with modern dark theme
- Added performance testing and metrics
- Fixed document chunking and retrieval

## Next Steps:

- [ ] Optimize chunk size for better retrieval
- [ ] Implement local embeddings using sentence-transformers
- [ ] Update deprecated LangChain imports
- [ ] Add batch processing for multiple documents
- [ ] Implement caching for faster responses

## Contributing:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
