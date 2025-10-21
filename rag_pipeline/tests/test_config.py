"""
Test configuration loading and management.

These tests verify that:
- Configuration can be loaded from YAML
- Paths are resolved correctly relative to base
- Dot-notation access works
- Missing keys return defaults
"""

import pytest
from pathlib import Path
from src.utils.config import Config, load_config


def test_config_loads_from_yaml():
    """Test that configuration can be loaded from YAML file."""
    config = load_config(Path("config/config.yaml"))
    
    # Verify key settings are loaded
    assert config.get("embeddings.model") == "all-MiniLM-L6-v2"
    assert config.get("retrieval.top_k") == 20
    assert config.get("chunking.strategies.fixed.chunk_size") == 512


def test_config_path_resolution():
    """Test that paths are resolved relative to project base."""
    config = load_config(Path("config/config.yaml"))
    
    # Get a path and verify it resolves correctly
    corpus_path = config.get_path("paths.corpus_root")
    # Path should contain the corpus directory name
    assert "data/corpus" in str(corpus_path) or "scikit-learn" in str(corpus_path)


def test_config_get_with_default():
    """Test that config.get() returns default for missing keys."""
    config = load_config(Path("config/config.yaml"))
    
    # Get non-existent key with default
    value = config.get("nonexistent.key", "default_value")
    assert value == "default_value"


def test_config_nested_key_access():
    """Test accessing nested configuration values."""
    config = load_config(Path("config/config.yaml"))
    
    # Access nested keys (note: config structure is chunking.strategies.fixed)
    assert config.get("chunking.strategies.fixed.enabled") is True
    assert config.get("chunking.strategies.semantic.max_chunk_size") == 1000
    assert config.get("generation.temperature") == 0.7

