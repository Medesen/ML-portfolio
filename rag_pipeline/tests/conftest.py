"""
Pytest configuration and shared fixtures.

Fixtures defined here are available to all test modules.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_document():
    """Provide a sample document for testing."""
    return {
        "doc_id": "test_doc_001",
        "title": "Test Document Title",
        "content": "This is a test document. " * 100,  # ~500 words
        "doc_type": "guide",
        "source_path": "test/path.html",
        "metadata": {
            "file_name": "test.html",
            "file_size": 1024,
            "sections": ["Section 1", "Section 2"]
        }
    }


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for testing."""
    return [
        {
            "chunk_id": "doc1_chunk_0",
            "doc_id": "doc1",
            "content": "This is the first chunk about StandardScaler.",
            "chunk_index": 0,
            "metadata": {"strategy": "fixed"}
        },
        {
            "chunk_id": "doc1_chunk_1",
            "doc_id": "doc1",
            "content": "This is the second chunk about preprocessing.",
            "chunk_index": 1,
            "metadata": {"strategy": "fixed"}
        },
        {
            "chunk_id": "doc2_chunk_0",
            "doc_id": "doc2",
            "content": "This chunk is from a different document about PCA.",
            "chunk_index": 0,
            "metadata": {"strategy": "fixed"}
        }
    ]


@pytest.fixture
def sample_test_question():
    """Provide a sample test question for evaluation testing."""
    return {
        "id": "q001",
        "question": "What does StandardScaler do?",
        "expected_topics": ["preprocessing", "standardization", "mean", "variance"],
        "relevant_doc_ids": ["doc1", "doc2"],
        "difficulty": "easy",
        "category": "factual"
    }

