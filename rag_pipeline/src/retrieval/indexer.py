"""Indexing orchestrator for building the vector database."""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..utils.config import Config
from ..utils.logger import get_logger
from ..chunking import FixedSizeChunker, SemanticChunker, HierarchicalChunker, Chunk
from .embedder import Embedder
from .vector_store import VectorStore


class IndexingStateTracker:
    """Track indexing state to avoid redundant operations."""
    
    def __init__(self, state_path: Path, vector_store_dir: Optional[Path] = None):
        """
        Initialize state tracker.
        
        Args:
            state_path: Path to state JSON file
            vector_store_dir: Path to vector store directory for validation
        """
        self.state_path = state_path
        self.vector_store_dir = vector_store_dir
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """Load state from file or return empty state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._empty_state()
        return self._empty_state()
    
    def _empty_state(self) -> Dict:
        """Return empty state structure."""
        return {
            "strategies": {}  # strategy_name -> {completed, timestamp, chunk_count, doc_count}
        }
    
    def save_state(self):
        """Save current state to file."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)
    
    def is_strategy_indexed(self, strategy_name: str, vector_store: VectorStore) -> bool:
        """
        Check if a strategy has been indexed.
        
        Validates that state says it's complete AND that the collection exists
        with the expected number of chunks.
        
        Args:
            strategy_name: Name of the chunking strategy
            vector_store: VectorStore instance for validation
            
        Returns:
            True if indexed and validated, False otherwise
        """
        # Check if state says it's complete
        if strategy_name not in self.state["strategies"]:
            return False
        
        strategy_state = self.state["strategies"][strategy_name]
        if not strategy_state.get("completed", False):
            return False
        
        # Validate that collection actually exists with chunks
        if vector_store:
            collection_info = vector_store.get_collection_info(strategy_name)
            if collection_info is None:
                # Collection doesn't exist - mark as incomplete
                self.state["strategies"][strategy_name]["completed"] = False
                self.save_state()
                return False
            
            # Check chunk count matches (exact match required)
            expected_count = strategy_state.get("chunk_count", 0)
            actual_count = collection_info.get("count", 0)
            
            if expected_count > 0 and actual_count != expected_count:
                # Mismatch - mark as incomplete
                self.state["strategies"][strategy_name]["completed"] = False
                self.save_state()
                return False
        
        return True
    
    def mark_strategy_completed(
        self, strategy_name: str, chunk_count: int, doc_count: int
    ):
        """Mark a strategy as indexed."""
        self.state["strategies"][strategy_name] = {
            "completed": True,
            "timestamp": datetime.utcnow().isoformat(),
            "chunk_count": chunk_count,
            "doc_count": doc_count,
        }
        self.save_state()
    
    def reset(self, strategy_name: Optional[str] = None):
        """
        Reset state.
        
        Args:
            strategy_name: If provided, reset only this strategy. Otherwise reset all.
        """
        if strategy_name:
            if strategy_name in self.state["strategies"]:
                del self.state["strategies"][strategy_name]
        else:
            self.state = self._empty_state()
        self.save_state()


class Indexer:
    """Orchestrate the indexing pipeline: chunk → embed → store."""
    
    def __init__(self, config: Config, logger_name: str = "indexer"):
        """
        Initialize indexer.
        
        Args:
            config: Configuration object
            logger_name: Name for the logger
        """
        self.config = config
        self.logger = get_logger(logger_name)
        
        # Setup paths
        self.processed_dir = config.get_path("paths.processed_dir")
        self.vector_store_dir = config.get_path("paths.vector_store_dir", create=True)
        self.state_dir = config.get_path("paths.state_dir", create=True)
        self.state_file = self.state_dir / "indexing_state.json"
        
        # Initialize components
        self.embedder = None  # Lazy initialization
        self.vector_store = VectorStore(self.vector_store_dir, logger_name="vector_store")
        self.state_tracker = IndexingStateTracker(self.state_file, self.vector_store_dir)
        
        # Initialize chunkers
        self.chunkers = self._initialize_chunkers()
    
    def _initialize_chunkers(self) -> Dict[str, Any]:
        """Initialize chunking strategies based on config."""
        chunkers = {}
        
        # Fixed-size chunker
        if self.config.get("chunking.strategies.fixed.enabled", False):
            fixed_config = self.config.get("chunking.strategies.fixed", {})
            chunkers["fixed"] = FixedSizeChunker(fixed_config)
            self.logger.info("Fixed-size chunker enabled")
        
        # Semantic chunker
        if self.config.get("chunking.strategies.semantic.enabled", False):
            semantic_config = self.config.get("chunking.strategies.semantic", {})
            chunkers["semantic"] = SemanticChunker(semantic_config)
            self.logger.info("Semantic chunker enabled")
        
        # Hierarchical chunker
        if self.config.get("chunking.strategies.hierarchical.enabled", False):
            hierarchical_config = self.config.get("chunking.strategies.hierarchical", {})
            chunkers["hierarchical"] = HierarchicalChunker(hierarchical_config)
            self.logger.info("Hierarchical chunker enabled")
        
        return chunkers
    
    def _get_embedder(self) -> Embedder:
        """Lazy initialization of embedder."""
        if self.embedder is None:
            model_name = self.config.get("embeddings.model", "all-MiniLM-L6-v2")
            device = self.config.get("embeddings.device", "cpu")
            batch_size = self.config.get("embeddings.batch_size", 32)
            
            self.embedder = Embedder(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                logger_name="embedder"
            )
        
        return self.embedder
    
    def index(
        self,
        strategy: Optional[str] = None,
        force_reindex: bool = False
    ):
        """
        Run the indexing pipeline.
        
        Args:
            strategy: Specific strategy to index ("fixed", "semantic", "hierarchical")
                     If None, index all enabled strategies
            force_reindex: Force re-indexing even if already done
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting INDEXING pipeline")
        self.logger.info("=" * 60)
        
        if force_reindex:
            self.logger.info("Force reindexing enabled")
            if strategy:
                self.state_tracker.reset(strategy)
            else:
                self.state_tracker.reset()
        
        # Determine which strategies to index
        if strategy:
            if strategy not in self.chunkers:
                raise ValueError(
                    f"Strategy '{strategy}' not enabled. "
                    f"Available: {list(self.chunkers.keys())}"
                )
            strategies_to_index = [strategy]
        else:
            strategies_to_index = list(self.chunkers.keys())
        
        if not strategies_to_index:
            self.logger.warning("No chunking strategies enabled in config")
            return
        
        # Load documents
        self.logger.info("Loading processed documents...")
        documents = self._load_documents()
        self.logger.info(f"Loaded {len(documents)} documents")
        
        # Index each strategy
        for strat_name in strategies_to_index:
            self._index_strategy(strat_name, documents, force_reindex)
        
        self.logger.info("=" * 60)
        self.logger.info("INDEXING pipeline completed")
        self.logger.info("=" * 60)
    
    def _index_strategy(
        self, strategy_name: str, documents: List[Dict], force_reindex: bool
    ):
        """Index documents with a specific chunking strategy."""
        # Check if already indexed
        if not force_reindex and self.state_tracker.is_strategy_indexed(
            strategy_name, self.vector_store
        ):
            self.logger.info(
                f"Strategy '{strategy_name}' already indexed (skipping)"
            )
            return
        
        self.logger.info(f"\nIndexing with strategy: {strategy_name}")
        self.logger.info("-" * 60)
        
        # Get chunker
        chunker = self.chunkers[strategy_name]
        
        # Chunk all documents
        self.logger.info("Chunking documents...")
        all_chunks = chunker.chunk_documents(documents)
        self.logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embedder = self._get_embedder()
        chunks_with_embeddings = embedder.embed_chunks(all_chunks, show_progress=True)
        
        # Store in vector database
        self.logger.info(f"Storing in vector database (collection: {strategy_name})...")
        embedding_dim = embedder.embedding_dim
        num_added = self.vector_store.add_chunks(
            collection_name=strategy_name,
            chunks=chunks_with_embeddings,
            embedding_dimension=embedding_dim
        )
        
        # Mark as completed
        self.state_tracker.mark_strategy_completed(
            strategy_name, len(all_chunks), len(documents)
        )
        
        self.logger.info(
            f"✓ Strategy '{strategy_name}' indexed successfully "
            f"({num_added} chunks from {len(documents)} documents)"
        )
    
    def _load_documents(self) -> List[Dict]:
        """Load all processed documents."""
        documents = []
        
        # Walk through all processed JSON files
        for json_file in self.processed_dir.rglob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    documents.append(doc)
            except Exception as e:
                self.logger.warning(f"Failed to load {json_file}: {e}")
        
        return documents
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        stats = {
            "vector_store": self.vector_store.get_stats(),
            "strategies": {},
        }
        
        for strategy_name in self.chunkers.keys():
            is_indexed = self.state_tracker.is_strategy_indexed(
                strategy_name, self.vector_store
            )
            strategy_state = self.state_tracker.state.get("strategies", {}).get(
                strategy_name, {}
            )
            
            stats["strategies"][strategy_name] = {
                "indexed": is_indexed,
                "chunk_count": strategy_state.get("chunk_count", 0),
                "doc_count": strategy_state.get("doc_count", 0),
                "timestamp": strategy_state.get("timestamp"),
            }
        
        return stats

