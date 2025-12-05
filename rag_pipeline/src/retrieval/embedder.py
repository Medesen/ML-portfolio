"""Embedding generation using sentence-transformers."""

from __future__ import annotations
from typing import List, Union, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logger import get_logger


class Embedder:
    """
    Wrapper for embedding generation using sentence-transformers.
    
    Designed to be modular - changing the embedding model only requires
    updating the model_name parameter or swapping the underlying library.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        logger_name: str = "embedder"
    ):
        """
        Initialize embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            logger_name: Logger name
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.logger = get_logger(logger_name)
        
        self.logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.logger.info(
            f"Model loaded. Embedding dimension: {self.embedding_dim}, Device: {device}"
        )
    
    def embed(
        self,
        texts: Union[str, List[str]],
        show_progress: bool = False,
        normalize: bool = True
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings to unit length
            
        Returns:
            Embedding array(s). If input is single string, returns single array.
            If input is list, returns list of arrays.
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        # Return single array for single input, list for multiple
        if single_input:
            return embeddings[0]
        return embeddings
    
    def embed_chunks(
        self,
        chunks: List[Any],  # List of Chunk objects
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Embed a list of Chunk objects and return them with embeddings.
        
        Args:
            chunks: List of Chunk objects (from chunking module)
            show_progress: Whether to show progress bar
            
        Returns:
            List of dictionaries with chunk data + embeddings
        """
        # Extract text content from chunks
        texts = [chunk.content for chunk in chunks]
        
        self.logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings
        embeddings = self.embed(texts, show_progress=show_progress)
        
        # Combine chunks with embeddings
        result = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_dict = chunk.to_dict()
            chunk_dict["embedding"] = embedding.tolist()  # Convert to list for JSON serialization
            result.append(chunk_dict)
        
        self.logger.info(f"Generated {len(result)} embeddings")
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
        }
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of recommended embedding models.
        
        Returns:
            List of model names
        """
        return [
            "all-MiniLM-L6-v2",  # Fast, good quality, 384 dims
            "all-mpnet-base-v2",  # Better quality, slower, 768 dims
            "multi-qa-MiniLM-L6-cos-v1",  # Optimized for Q&A
            "paraphrase-MiniLM-L6-v2",  # Good for paraphrase detection
        ]

