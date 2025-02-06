from sentence_transformers import SentenceTransformer
import torch
import logging
import time
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class LocalEmbeddings(Embeddings):
    """Local embedding model using sentence-transformers."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a lightweight model.
        
        all-MiniLM-L6-v2 is:
        - 6 layers (vs 12 in base BERT)
        - 384 embedding dimensions (vs 768 in base BERT)
        - 22MB model size
        - Excellent speed/performance trade-off
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        
        start_time = time.time()
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        logger.info(f"Model loaded in {time.time() - start_time:.2f}s")
        
        # Cache for query embeddings
        self._query_cache = {}
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""
        if not texts:
            return []
        
        start_time = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False,  # Return numpy array directly
            device=self.device
        )
        
        logger.info(f"Embedded {len(texts)} documents in {time.time() - start_time:.2f}s")
        return embeddings.tolist()  # Convert to list for storage
    
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query with caching."""
        if text in self._query_cache:
            return self._query_cache[text]
        
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_tensor=False,
            device=self.device
        )
        
        self._query_cache[text] = embedding.tolist()
        return self._query_cache[text]
