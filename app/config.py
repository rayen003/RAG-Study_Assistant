import os
from dotenv import load_dotenv
from pathlib import Path

# Get the absolute path to the .env file
env_path = Path(__file__).parent.parent / '.env'

# Load environment variables from .env file
load_dotenv(dotenv_path=env_path)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file with your API key.")

# Model configurations
MODEL_NAME = "gpt-4-1106-preview"  # Using GPT-4 for better comprehension
MODEL_TEMPERATURE = 0.7  # Higher temperature for more detailed and creative responses
EMBEDDING_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 1000

# File storage configuration
UPLOAD_FOLDER = Path(__file__).parent.parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)