import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info(f"Successfully loaded embedding model on {self.device}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            logger.debug(f"Generated embedding of shape {embedding.shape}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> List[np.ndarray]:
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[np.ndarray]:
    embedder = TextEmbedder(model_name=model_name)
    return embedder.embed_batch(texts)


if __name__ == "__main__":
    sample_texts = [
        "This is the first sample text about computer science.",
        "This is the second sample text about engineering programs.",
        "Department of Electrical Engineering offers various programs."
    ]
    
    embedder = TextEmbedder()
    print(f"Embedding dimension: {embedder.get_embedding_dimension()}")
    
    embeddings = embedder.embed_batch(sample_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"First embedding shape: {embeddings[0].shape}")
