"""Query processing and retrieval interface."""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Union
import time
from pathlib import Path

from .embedder import Embedder
from .vector_store import VectorStore
from ..utils.logger import get_logger


class QueryProcessor:
    """
    Query processor for RAG pipeline.
    
    Handles query preprocessing, embedding generation, vector retrieval,
    and result formatting. Supports querying single or multiple chunking
    strategies with result merging.
    """
    
    def __init__(
        self,
        config,
        embedder: Embedder,
        vector_store: VectorStore,
        logger_name: str = "query_processor"
    ):
        """
        Initialize query processor.
        
        Args:
            config: Configuration object
            embedder: Embedder instance for generating query embeddings
            vector_store: VectorStore instance for retrieval
            logger_name: Logger name
        """
        self.config = config
        self.embedder = embedder
        self.vector_store = vector_store
        self.logger = get_logger(logger_name)
        
        # Load retrieval configuration
        self.default_top_k = config.get("retrieval.top_k", 20)
        self.min_similarity = config.get("retrieval.min_similarity", 0.0)
        self.merge_strategy = config.get("retrieval.merge_strategy", "interleave")
        self.deduplicate = config.get("retrieval.deduplicate", True)
        
        self.logger.info("Query processor initialized")
    
    def process_query(
        self,
        query_text: str,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        show_full_content: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query and retrieve relevant chunks.
        
        Args:
            query_text: Natural language query
            strategy: Chunking strategy to query ("fixed", "semantic", "hierarchical", "all", or None for default)
            top_k: Number of results to return (None for default)
            filters: Metadata filters for ChromaDB (e.g., {"doc_type": "guide"})
            show_full_content: Whether to include full chunk content
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        
        # Normalize query
        query_text = self._normalize_query(query_text)
        self.logger.info(f"Processing query: '{query_text}'")
        
        # Determine strategy
        if strategy is None:
            strategy = self.config.get("retrieval.strategy", "all")
        
        # Determine top_k
        if top_k is None:
            top_k = self.default_top_k
        
        # Generate query embedding
        self.logger.info("Generating query embedding...")
        embed_start = time.time()
        query_embedding = self.embedder.embed(query_text, show_progress=False, normalize=True)
        embed_time = time.time() - embed_start
        self.logger.info(f"Query embedding generated in {embed_time:.3f}s")
        
        # Retrieve from appropriate strategy/strategies
        retrieval_start = time.time()
        if strategy == "all":
            results = self._query_all_strategies(
                query_embedding, top_k, filters
            )
        else:
            results = self._query_single_strategy(
                query_embedding, strategy, top_k, filters
            )
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        # Format results
        formatted_results = {
            "query": query_text,
            "strategy": strategy,
            "results": results,
            "metadata": {
                "total_results": len(results),
                "top_k_requested": top_k,
                "min_similarity": self.min_similarity,
                "filters": filters,
                "timing": {
                    "embedding_time": round(embed_time, 3),
                    "retrieval_time": round(retrieval_time, 3),
                    "total_time": round(total_time, 3)
                }
            }
        }
        
        # Add strategy info if querying all
        if strategy == "all":
            strategies_queried = list(set(r["strategy"] for r in results))
            formatted_results["metadata"]["strategies_queried"] = strategies_queried
        
        self.logger.info(
            f"Query completed: {len(results)} results in {total_time:.3f}s"
        )
        
        return formatted_results
    
    def _normalize_query(self, query_text: str) -> str:
        """
        Normalize query text.
        
        Args:
            query_text: Raw query text
            
        Returns:
            Normalized query text
        """
        # Basic normalization: strip whitespace
        query_text = query_text.strip()
        
        # Remove extra whitespace
        query_text = " ".join(query_text.split())
        
        return query_text
    
    def _query_single_strategy(
        self,
        query_embedding: List[float],
        strategy: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query a single chunking strategy.
        
        Args:
            query_embedding: Query embedding vector
            strategy: Strategy name
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of result dictionaries
        """
        self.logger.info(f"Querying strategy: {strategy} (top_k={top_k})")
        
        # Check if collection exists
        collection_info = self.vector_store.get_collection_info(strategy)
        if collection_info is None:
            self.logger.warning(
                f"Strategy '{strategy}' not found or not indexed. "
                f"Run 'index --strategy {strategy}' first."
            )
            return []
        
        # Query vector store
        raw_results = self.vector_store.query(
            collection_name=strategy,
            query_embedding=query_embedding.tolist(),
            n_results=top_k,
            where=filters
        )
        
        # Format results
        results = self._format_results(raw_results, strategy)
        
        # Filter by minimum similarity
        if self.min_similarity > 0.0:
            results = [
                r for r in results
                if r["similarity_score"] >= self.min_similarity
            ]
        
        self.logger.info(f"Retrieved {len(results)} results from '{strategy}'")
        return results
    
    def _query_all_strategies(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query all enabled chunking strategies and merge results.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results (total across all strategies)
            filters: Metadata filters
            
        Returns:
            List of merged result dictionaries
        """
        self.logger.info(f"Querying all strategies (top_k={top_k})")
        
        # Get list of available strategies
        available_collections = self.vector_store.list_collections()
        strategies = ["fixed", "semantic", "hierarchical"]
        strategies = [s for s in strategies if s in available_collections]
        
        if not strategies:
            self.logger.warning(
                "No indexed strategies found. Run 'index' command first."
            )
            return []
        
        self.logger.info(f"Found {len(strategies)} strategies: {strategies}")
        
        # Query each strategy
        all_results = []
        for strategy in strategies:
            # Query with higher top_k (2x requested) to ensure we have enough results
            # after merging and deduplication. Example: if top_k=20, we get 40 from each
            # strategy, then merge and deduplicate down to best 20 overall.
            strategy_results = self._query_single_strategy(
                query_embedding, strategy, top_k * 2, filters
            )
            all_results.extend(strategy_results)
        
        # Merge results
        merged_results = self._merge_results(all_results, top_k)
        
        self.logger.info(
            f"Merged {len(all_results)} results from {len(strategies)} strategies "
            f"into top {len(merged_results)}"
        )
        
        return merged_results
    
    def _format_results(
        self,
        raw_results: Dict[str, Any],
        strategy_name: str
    ) -> List[Dict[str, Any]]:
        """
        Format ChromaDB results into readable format.
        
        Args:
            raw_results: Raw ChromaDB query results
            strategy_name: Name of the strategy
            
        Returns:
            List of formatted result dictionaries
        """
        formatted = []
        
        # ChromaDB returns results as lists within the dictionary
        ids = raw_results.get("ids", [[]])[0]
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]
        
        for i, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            # Convert distance to similarity score (ChromaDB uses L2 distance)
            # For normalized vectors: similarity = 1 - (distance^2 / 2)
            # Simplified: similarity ≈ 1 - distance/2 (good approximation)
            similarity_score = max(0.0, 1.0 - (distance / 2.0))
            
            result = {
                "rank": i + 1,
                "chunk_id": chunk_id,
                "doc_id": metadata.get("doc_id", "unknown"),
                "content": content,
                "similarity_score": round(similarity_score, 4),
                "metadata": metadata,
                "strategy": strategy_name
            }
            
            formatted.append(result)
        
        return formatted
    
    def _merge_results(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Merge results from multiple strategies.
        
        Args:
            results: List of results from different strategies
            top_k: Number of results to return
            
        Returns:
            Merged and ranked results
        """
        if self.merge_strategy == "top_scores":
            # Top scores strategy: Sort all results by similarity regardless of strategy
            # This maximizes relevance but may favor strategies that produce higher scores
            merged = sorted(
                results,
                key=lambda x: x["similarity_score"],
                reverse=True
            )
        else:  # "interleave"
            # Interleave strategy: Round-robin from each strategy to ensure balanced representation
            # Example with 3 strategies: [Fixed_1, Semantic_1, Hier_1, Fixed_2, Semantic_2, ...]
            # This prevents one strategy from dominating results
            
            # Group results by their chunking strategy
            by_strategy = {}
            for result in results:
                strategy = result["strategy"]
                if strategy not in by_strategy:
                    by_strategy[strategy] = []
                by_strategy[strategy].append(result)
            
            # Sort each strategy's results by similarity score (best first)
            for strategy in by_strategy:
                by_strategy[strategy].sort(
                    key=lambda x: x["similarity_score"],
                    reverse=True
                )
            
            # Interleave results
            merged = []
            strategy_names = list(by_strategy.keys())
            max_per_strategy = max(len(r) for r in by_strategy.values())
            
            for i in range(max_per_strategy):
                for strategy in strategy_names:
                    if i < len(by_strategy[strategy]):
                        merged.append(by_strategy[strategy][i])
                    if len(merged) >= top_k:
                        break
                if len(merged) >= top_k:
                    break
        
        # Deduplicate if enabled (same doc_id + similar content)
        if self.deduplicate:
            merged = self._deduplicate_results(merged)
        
        # Take top_k
        merged = merged[:top_k]
        
        # Re-rank
        for i, result in enumerate(merged):
            result["rank"] = i + 1
        
        return merged
    
    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate results (same doc_id appearing multiple times).
        
        Keeps the result with highest similarity score for each doc_id.
        
        Args:
            results: List of results
            
        Returns:
            Deduplicated results
        """
        seen_docs = {}
        
        for result in results:
            doc_id = result["doc_id"]
            
            if doc_id not in seen_docs:
                seen_docs[doc_id] = result
            else:
                # Keep the one with higher similarity
                if result["similarity_score"] > seen_docs[doc_id]["similarity_score"]:
                    seen_docs[doc_id] = result
        
        # Convert back to list and sort by similarity
        deduplicated = list(seen_docs.values())
        deduplicated.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return deduplicated
    
    def format_console_output(
        self,
        results: Dict[str, Any],
        show_full_content: bool = False,
        max_excerpt_length: int = 200
    ) -> str:
        """
        Format results for console display.
        
        Args:
            results: Query results dictionary
            show_full_content: Whether to show full content
            max_excerpt_length: Maximum length for content excerpts
            
        Returns:
            Formatted string for console output
        """
        lines = []
        lines.append("=" * 80)
        lines.append("QUERY RESULTS")
        lines.append("=" * 80)
        lines.append(f"Query: \"{results['query']}\"")
        lines.append(f"Strategy: {results['strategy']}")
        lines.append(f"Results: {results['metadata']['total_results']}")
        
        timing = results['metadata']['timing']
        lines.append(
            f"Time: {timing['total_time']}s "
            f"(embedding: {timing['embedding_time']}s, "
            f"retrieval: {timing['retrieval_time']}s)"
        )
        
        if results['metadata'].get('strategies_queried'):
            strategies = ", ".join(results['metadata']['strategies_queried'])
            lines.append(f"Strategies queried: {strategies}")
        
        lines.append("-" * 80)
        
        # Display results
        if not results['results']:
            lines.append("\nNo results found.")
        else:
            for result in results['results']:
                lines.append(f"\nRank {result['rank']} [{result['strategy']}] "
                           f"(similarity: {result['similarity_score']:.4f})")
                lines.append(f"Doc: {result['doc_id']}")
                lines.append(f"Chunk: {result['chunk_id']}")
                
                # Content display
                content = result['content']
                if show_full_content:
                    lines.append(f"\n{content}")
                else:
                    excerpt = self._create_excerpt(content, max_excerpt_length)
                    lines.append(f"\n{excerpt}")
                
                lines.append("")  # Blank line between results
        
        lines.append("-" * 80)
        lines.append(f"Retrieved {results['metadata']['total_results']} results")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _create_excerpt(self, text: str, max_length: int) -> str:
        """
        Create an excerpt from text.
        
        Args:
            text: Full text
            max_length: Maximum length
            
        Returns:
            Truncated text with ellipsis
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at sentence boundary
        excerpt = text[:max_length]
        last_period = excerpt.rfind(". ")
        
        if last_period > max_length * 0.7:  # At least 70% of max_length
            return excerpt[:last_period + 1] + ".."
        
        # Otherwise, break at word boundary
        last_space = excerpt.rfind(" ")
        if last_space > 0:
            return excerpt[:last_space] + "..."
        
        return excerpt + "..."

