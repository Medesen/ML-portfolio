"""Base abstract class for document chunking strategies."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Any


class Chunk:
    """Represents a single chunk of text with metadata."""
    
    def __init__(
        self,
        content: str,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        metadata: Dict[str, Any] = None,
    ):
        """
        Initialize a chunk.
        
        Args:
            content: The text content of the chunk
            chunk_id: Unique identifier for this chunk
            doc_id: ID of the source document
            chunk_index: Position of this chunk in the document (0-indexed)
            metadata: Additional metadata (source, section, etc.)
        """
        self.content = content
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary representation."""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
    
    def __repr__(self) -> str:
        """String representation of chunk."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.chunk_id}, doc={self.doc_id}, idx={self.chunk_index}, preview='{preview}')"


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Configuration dictionary for the chunker
        """
        self.config = config or {}
        
    @abstractmethod
    def chunk_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a single document into smaller pieces.
        
        Args:
            document: Document dictionary with 'doc_id', 'content', and metadata
            
        Returns:
            List of Chunk objects
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this chunking strategy.
        
        Returns:
            Strategy name (e.g., 'fixed', 'semantic', 'hierarchical')
        """
        pass
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
    
    def _create_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """
        Create a unique chunk ID.
        
        Args:
            doc_id: Document ID
            chunk_index: Index of the chunk
            
        Returns:
            Unique chunk identifier
        """
        return f"{doc_id}__chunk_{chunk_index}"
    
    def _extract_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant metadata from document.
        
        Args:
            document: Document dictionary
            
        Returns:
            Metadata dictionary
        """
        return {
            "source_path": document.get("source_path"),
            "doc_type": document.get("doc_type"),
            "title": document.get("title"),
            "strategy": self.get_strategy_name(),
        }

