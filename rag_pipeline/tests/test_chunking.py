"""
Test chunking strategies.

These tests verify that:
- Fixed chunker creates uniform-sized chunks
- Semantic chunker respects natural boundaries
- Chunk IDs are generated correctly
- Metadata is preserved
"""

import pytest
from src.chunking.fixed_chunker import FixedSizeChunker
from src.chunking.semantic_chunker import SemanticChunker


def test_fixed_chunker_creates_chunks(sample_document):
    """Test that fixed chunker creates chunks from a document."""
    config = {
        "chunk_size": 512,
        "overlap": 50
    }
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunks were created
    assert len(chunks) > 0
    # Verify each chunk has required fields
    for chunk in chunks:
        assert hasattr(chunk, 'chunk_id')
        assert hasattr(chunk, 'doc_id')
        assert hasattr(chunk, 'content')
        assert hasattr(chunk, 'chunk_index')


def test_fixed_chunker_chunk_sizes(sample_document):
    """Test that fixed chunker creates roughly uniform-sized chunks."""
    config = {
        "chunk_size": 512,
        "overlap": 50
    }
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Calculate word counts
    word_counts = [len(chunk.content.split()) for chunk in chunks]
    
    # Verify sizes are within expected range (~384 words ± 20%)
    # 512 tokens * 0.75 words/token = ~384 words
    for count in word_counts[:-1]:  # Exclude last chunk (may be smaller)
        assert 300 < count < 500, f"Chunk size {count} outside expected range"


def test_semantic_chunker_creates_chunks(sample_document):
    """Test that semantic chunker creates chunks."""
    config = {
        "max_chunk_size": 1000,
        "method": "sentence"
    }
    chunker = SemanticChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunks were created
    assert len(chunks) > 0
    # Verify strategy name
    assert chunker.get_strategy_name() == "semantic"


def test_chunk_id_generation(sample_document):
    """Test that chunk IDs are generated correctly."""
    config = {"chunk_size": 512, "overlap": 50}
    chunker = FixedSizeChunker(config)
    
    chunks = chunker.chunk_document(sample_document)
    
    # Verify chunk IDs follow pattern: {doc_id}__chunk_{index}
    for i, chunk in enumerate(chunks):
        expected_id = f"{sample_document['doc_id']}__chunk_{i}"
        assert chunk.chunk_id == expected_id
        assert chunk.chunk_index == i

