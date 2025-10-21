"""
Test embedder functionality.

These tests verify that:
- Embedder can be initialized
- Embeddings have correct dimensions
- Batch processing works
- Model info is accessible

Note: These tests use the actual sentence-transformers model.
In production, you might mock the model for faster tests.
"""

import pytest
import numpy as np
from src.retrieval.embedder import Embedder


@pytest.fixture
def embedder():
    """Create an embedder instance for testing."""
    # Use the default model from config
    return Embedder(
        model_name="all-MiniLM-L6-v2",
        device="cpu",
        batch_size=2
    )


def test_embedder_initialization(embedder):
    """Test that embedder initializes correctly."""
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.device == "cpu"
    assert embedder.batch_size == 2


def test_embedder_single_text(embedder):
    """Test embedding a single text."""
    text = "This is a test sentence."
    
    embedding = embedder.embed(text, show_progress=False)
    
    # Verify embedding shape (384 dimensions for all-MiniLM-L6-v2)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)


def test_embedder_multiple_texts(embedder):
    """Test embedding multiple texts."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    
    embeddings = embedder.embed(texts, show_progress=False)
    
    # Verify embeddings shape (3 texts × 384 dimensions)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)


def test_embedder_normalization(embedder):
    """Test that embeddings are normalized when requested."""
    text = "Test sentence for normalization."
    
    embedding = embedder.embed(text, normalize=True, show_progress=False)
    
    # Verify L2 norm is approximately 1.0 (normalized)
    norm = np.linalg.norm(embedding)
    assert abs(norm - 1.0) < 0.001

