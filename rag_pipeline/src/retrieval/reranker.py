"""Cross-encoder reranker for improving retrieval quality."""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import time

from sentence_transformers import CrossEncoder

from ..utils.logger import get_logger


class CrossEncoderReranker:
    """
    Reranks search results using a cross-encoder model.
    
    Cross-encoders jointly encode query-document pairs and produce
    a relevance score, providing more accurate relevance estimates
    than bi-encoder similarity at the cost of higher latency.
    
    Typical usage:
    1. Over-fetch ~50 documents from initial retrieval (BM25 + semantic)
    2. Rerank using cross-encoder
    3. Return top 10 after reranking
    
    Note: The first call to rerank() may be slower due to model loading.
    Subsequent calls will be faster as the model remains in memory.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        logger_name: str = "reranker"
    ):
        """
        Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder.
                        Default: "cross-encoder/ms-marco-MiniLM-L-6-v2"
            device: Device to run model on ("cpu" or "cuda")
            batch_size: Batch size for scoring query-document pairs
            logger_name: Logger name
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.logger = get_logger(logger_name)
        
        self.model: Optional[CrossEncoder] = None
        self._model_load_time: Optional[float] = None
        
        self.logger.info(
            f"CrossEncoderReranker initialized (model={model_name}, device={device})"
        )
    
    def _ensure_model_loaded(self) -> bool:
        """
        Lazy-load the cross-encoder model.
        
        Returns:
            True if model is loaded successfully, False otherwise
        """
        if self.model is not None:
            return True
        
        try:
            self.logger.info(f"Loading cross-encoder model: {self.model_name}")
            load_start = time.time()
            
            self.model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            
            self._model_load_time = time.time() - load_start
            self.logger.info(
                f"Cross-encoder model loaded in {self._model_load_time:.2f}s"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder model: {e}")
            return False
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Rerank results using cross-encoder scoring.
        
        Args:
            query: The search query
            results: List of result dicts, each must have a 'content' field
            top_k: Number of results to return after reranking
            
        Returns:
            Dictionary with:
                - 'results': Reranked and truncated results list
                - 'rerank_time_ms': Time taken for reranking in milliseconds
                - 'reranked': Boolean indicating if reranking was performed
        """
        rerank_start = time.time()
        
        # Handle empty results
        if not results:
            return {
                'results': [],
                'rerank_time_ms': 0.0,
                'reranked': False
            }
        
        # Ensure model is loaded
        if not self._ensure_model_loaded():
            # Fallback: return top_k unreranked results
            self.logger.warning(
                "Reranking failed (model not loaded). "
                f"Returning top {top_k} unreranked results."
            )
            fallback_results = results[:top_k]
            # Update ranks for fallback results
            for i, result in enumerate(fallback_results, start=1):
                result['rank'] = i
            
            return {
                'results': fallback_results,
                'rerank_time_ms': round((time.time() - rerank_start) * 1000, 2),
                'reranked': False
            }
        
        try:
            # Create query-document pairs for scoring
            pairs = [(query, result['content']) for result in results]
            
            # Score all pairs using cross-encoder
            self.logger.debug(f"Scoring {len(pairs)} query-document pairs")
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Add rerank scores to results
            for result, score in zip(results, scores):
                result['rerank_score'] = float(score)
            
            # Sort by rerank score (descending)
            reranked_results = sorted(
                results,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            # Truncate to top_k and update ranks
            final_results = reranked_results[:top_k]
            for i, result in enumerate(final_results, start=1):
                result['rank'] = i
            
            rerank_time_ms = round((time.time() - rerank_start) * 1000, 2)
            
            self.logger.info(
                f"Reranked {len(results)} results to top {len(final_results)} "
                f"in {rerank_time_ms:.1f}ms"
            )
            
            return {
                'results': final_results,
                'rerank_time_ms': rerank_time_ms,
                'reranked': True
            }
            
        except Exception as e:
            # Fallback: return top_k unreranked results
            self.logger.warning(
                f"Reranking failed: {e}. "
                f"Returning top {top_k} unreranked results."
            )
            fallback_results = results[:top_k]
            # Update ranks for fallback results
            for i, result in enumerate(fallback_results, start=1):
                result['rank'] = i
            
            return {
                'results': fallback_results,
                'rerank_time_ms': round((time.time() - rerank_start) * 1000, 2),
                'reranked': False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get reranker statistics and configuration.
        
        Returns:
            Dictionary with reranker stats
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'model_loaded': self.model is not None,
            'model_load_time': self._model_load_time
        }
