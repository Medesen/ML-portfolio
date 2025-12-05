"""Tests for cross-encoder reranker functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker class."""
    
    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing."""
        return [
            {
                'chunk_id': 'c1',
                'doc_id': 'd1',
                'content': 'StandardScaler normalizes features by removing the mean.',
                'rrf_score': 0.016,
                'rank': 1,
                'metadata': {'source': 'preprocessing.html'}
            },
            {
                'chunk_id': 'c2',
                'doc_id': 'd1',
                'content': 'Use fit_transform to fit and transform data.',
                'rrf_score': 0.015,
                'rank': 2,
                'metadata': {'source': 'preprocessing.html'}
            },
            {
                'chunk_id': 'c3',
                'doc_id': 'd2',
                'content': 'GridSearchCV performs hyperparameter tuning.',
                'rrf_score': 0.014,
                'rank': 3,
                'metadata': {'source': 'model_selection.html'}
            },
            {
                'chunk_id': 'c4',
                'doc_id': 'd3',
                'content': 'PCA reduces dimensionality by projecting data.',
                'rrf_score': 0.013,
                'rank': 4,
                'metadata': {'source': 'decomposition.html'}
            },
            {
                'chunk_id': 'c5',
                'doc_id': 'd4',
                'content': 'RandomForestClassifier is an ensemble method.',
                'rrf_score': 0.012,
                'rank': 5,
                'metadata': {'source': 'ensemble.html'}
            },
        ]
    
    @pytest.fixture
    def mock_cross_encoder(self):
        """Create a mock CrossEncoder model."""
        mock = MagicMock()
        # Return scores that reorder: c3 best, then c1, c5, c2, c4
        mock.predict.return_value = np.array([0.7, 0.5, 0.9, 0.3, 0.6])
        return mock
    
    @pytest.fixture
    def reranker_with_mock(self, mock_cross_encoder):
        """Create a reranker with mocked CrossEncoder."""
        with patch('src.retrieval.reranker.CrossEncoder', return_value=mock_cross_encoder):
            reranker = CrossEncoderReranker(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu",
                batch_size=32
            )
            # Force model loading
            reranker._ensure_model_loaded()
            return reranker
    
    def test_initialization(self):
        """Test reranker initializes correctly without loading model."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=32
        )
        
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device == "cpu"
        assert reranker.batch_size == 32
        assert reranker.model is None  # Lazy loading
    
    def test_rerank_reorders_by_score(self, reranker_with_mock, sample_results):
        """Test that reranking reorders results by cross-encoder score."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        reranked = result['results']
        
        # Verify reordering: c3 (0.9), c1 (0.7), c5 (0.6), c2 (0.5), c4 (0.3)
        assert reranked[0]['chunk_id'] == 'c3'
        assert reranked[1]['chunk_id'] == 'c1'
        assert reranked[2]['chunk_id'] == 'c5'
        assert reranked[3]['chunk_id'] == 'c2'
        assert reranked[4]['chunk_id'] == 'c4'
    
    def test_rerank_adds_rerank_score(self, reranker_with_mock, sample_results):
        """Test that reranking adds rerank_score to results."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        for r in result['results']:
            assert 'rerank_score' in r
            assert isinstance(r['rerank_score'], float)
    
    def test_rerank_truncates_to_top_k(self, reranker_with_mock, sample_results):
        """Test that reranking truncates results to top_k."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=3
        )
        
        assert len(result['results']) == 3
        # Should be top 3 by rerank score: c3, c1, c5
        assert result['results'][0]['chunk_id'] == 'c3'
        assert result['results'][1]['chunk_id'] == 'c1'
        assert result['results'][2]['chunk_id'] == 'c5'
    
    def test_rerank_updates_ranks(self, reranker_with_mock, sample_results):
        """Test that reranking updates rank field correctly."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        for i, r in enumerate(result['results'], start=1):
            assert r['rank'] == i
    
    def test_rerank_preserves_metadata(self, reranker_with_mock, sample_results):
        """Test that reranking preserves chunk metadata."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        # Find c3 in results (should be first)
        c3_result = result['results'][0]
        assert c3_result['chunk_id'] == 'c3'
        assert c3_result['doc_id'] == 'd2'
        assert c3_result['content'] == 'GridSearchCV performs hyperparameter tuning.'
        assert c3_result['metadata']['source'] == 'model_selection.html'
    
    def test_rerank_returns_timing_metadata(self, reranker_with_mock, sample_results):
        """Test that reranking returns timing metadata."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        assert 'rerank_time_ms' in result
        assert isinstance(result['rerank_time_ms'], float)
        assert result['rerank_time_ms'] >= 0
    
    def test_rerank_returns_reranked_flag(self, reranker_with_mock, sample_results):
        """Test that reranking returns reranked boolean flag."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=sample_results,
            top_k=5
        )
        
        assert 'reranked' in result
        assert result['reranked'] is True
    
    def test_rerank_empty_results(self, reranker_with_mock):
        """Test reranking with empty results list."""
        result = reranker_with_mock.rerank(
            query="How to normalize features?",
            results=[],
            top_k=5
        )
        
        assert result['results'] == []
        assert result['rerank_time_ms'] == 0.0
        assert result['reranked'] is False
    
    def test_rerank_fewer_results_than_top_k(self, reranker_with_mock):
        """Test reranking when results count is less than top_k."""
        results = [
            {'chunk_id': 'c1', 'doc_id': 'd1', 'content': 'Content 1', 'rank': 1},
            {'chunk_id': 'c2', 'doc_id': 'd2', 'content': 'Content 2', 'rank': 2},
        ]
        
        # Mock returns scores for 2 results
        reranker_with_mock.model.predict.return_value = np.array([0.3, 0.8])
        
        result = reranker_with_mock.rerank(
            query="test query",
            results=results,
            top_k=10  # Request more than available
        )
        
        assert len(result['results']) == 2
        assert result['results'][0]['chunk_id'] == 'c2'  # Higher score
        assert result['results'][1]['chunk_id'] == 'c1'
    
    def test_fallback_on_model_load_failure(self, sample_results):
        """Test fallback behavior when model fails to load."""
        with patch('src.retrieval.reranker.CrossEncoder', side_effect=Exception("Model load failed")):
            reranker = CrossEncoderReranker(
                model_name="invalid-model",
                device="cpu"
            )
            
            result = reranker.rerank(
                query="test query",
                results=sample_results,
                top_k=3
            )
            
            # Should return top 3 unreranked results
            assert len(result['results']) == 3
            assert result['reranked'] is False
            # Original order preserved (by original rank)
            assert result['results'][0]['chunk_id'] == 'c1'
            assert result['results'][1]['chunk_id'] == 'c2'
            assert result['results'][2]['chunk_id'] == 'c3'
    
    def test_fallback_on_predict_failure(self, reranker_with_mock, sample_results):
        """Test fallback behavior when predict() fails."""
        # Make predict raise an exception
        reranker_with_mock.model.predict.side_effect = Exception("Prediction failed")
        
        result = reranker_with_mock.rerank(
            query="test query",
            results=sample_results,
            top_k=3
        )
        
        # Should return top 3 unreranked results
        assert len(result['results']) == 3
        assert result['reranked'] is False
        # Original order preserved
        assert result['results'][0]['chunk_id'] == 'c1'
        assert result['results'][1]['chunk_id'] == 'c2'
        assert result['results'][2]['chunk_id'] == 'c3'
    
    def test_fallback_updates_ranks(self, sample_results):
        """Test that fallback updates rank field correctly."""
        with patch('src.retrieval.reranker.CrossEncoder', side_effect=Exception("Model load failed")):
            reranker = CrossEncoderReranker(model_name="invalid-model")
            
            result = reranker.rerank(
                query="test query",
                results=sample_results,
                top_k=3
            )
            
            # Ranks should be updated to 1, 2, 3
            for i, r in enumerate(result['results'], start=1):
                assert r['rank'] == i
    
    def test_get_stats_before_model_load(self):
        """Test get_stats before model is loaded."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cuda",
            batch_size=64
        )
        
        stats = reranker.get_stats()
        
        assert stats['model_name'] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert stats['device'] == "cuda"
        assert stats['batch_size'] == 64
        assert stats['model_loaded'] is False
        assert stats['model_load_time'] is None
    
    def test_get_stats_after_model_load(self, reranker_with_mock):
        """Test get_stats after model is loaded."""
        stats = reranker_with_mock.get_stats()
        
        assert stats['model_loaded'] is True
        assert stats['model_load_time'] is not None
    
    def test_lazy_loading(self, mock_cross_encoder):
        """Test that model is lazily loaded on first rerank call."""
        with patch('src.retrieval.reranker.CrossEncoder', return_value=mock_cross_encoder) as mock_class:
            reranker = CrossEncoderReranker()
            
            # Model should not be loaded yet
            assert reranker.model is None
            mock_class.assert_not_called()
            
            # First rerank triggers loading
            results = [{'chunk_id': 'c1', 'content': 'test', 'rank': 1}]
            mock_cross_encoder.predict.return_value = np.array([0.5])
            
            reranker.rerank("test query", results, top_k=1)
            
            mock_class.assert_called_once()
            assert reranker.model is not None


class TestCrossEncoderRerankerIntegration:
    """Integration tests for reranker with HybridSearcher."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for HybridSearcher."""
        mock_vector_store = Mock()
        mock_vector_store.query.return_value = {
            'ids': [['c1', 'c2', 'c3']],
            'documents': [['Doc 1', 'Doc 2', 'Doc 3']],
            'metadatas': [[{'doc_id': 'd1'}, {'doc_id': 'd2'}, {'doc_id': 'd3'}]],
            'distances': [[0.2, 0.4, 0.6]]
        }
        mock_vector_store.list_collections.return_value = ['fixed']
        
        mock_bm25 = Mock()
        mock_bm25.search.return_value = [
            {'chunk_id': 'c2', 'doc_id': 'd2', 'content': 'Doc 2', 'bm25_score': 5.0, 'bm25_rank': 1},
            {'chunk_id': 'c1', 'doc_id': 'd1', 'content': 'Doc 1', 'bm25_score': 3.0, 'bm25_rank': 2},
        ]
        mock_bm25.get_stats.return_value = {'num_documents': 3}
        
        mock_embedder = Mock()
        mock_embedder.embed.return_value = np.zeros(384)
        
        return mock_vector_store, mock_bm25, mock_embedder
    
    def test_hybrid_searcher_with_reranker(self, mock_components):
        """Test HybridSearcher integration with reranker."""
        from src.retrieval.hybrid_searcher import HybridSearcher
        
        mock_vector_store, mock_bm25, mock_embedder = mock_components
        
        # Create mock reranker
        mock_reranker = Mock()
        mock_reranker.rerank.return_value = {
            'results': [
                {'chunk_id': 'c2', 'rank': 1, 'rerank_score': 0.9},
                {'chunk_id': 'c1', 'rank': 2, 'rerank_score': 0.7},
            ],
            'rerank_time_ms': 15.5,
            'reranked': True
        }
        
        searcher = HybridSearcher(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25,
            embedder=mock_embedder,
            reranker=mock_reranker
        )
        
        result = searcher.search(
            query="test query",
            strategy="fixed",
            top_k=10,
            overfetch_k=50,
            rerank_top_k=2
        )
        
        # Verify reranker was called
        mock_reranker.rerank.assert_called_once()
        
        # Verify results are from reranker
        assert len(result['results']) == 2
        assert result['metadata']['reranked'] is True
        assert result['metadata']['timing']['rerank_ms'] == 15.5
    
    def test_hybrid_searcher_without_reranker(self, mock_components):
        """Test HybridSearcher works without reranker."""
        from src.retrieval.hybrid_searcher import HybridSearcher
        
        mock_vector_store, mock_bm25, mock_embedder = mock_components
        
        searcher = HybridSearcher(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25,
            embedder=mock_embedder,
            reranker=None
        )
        
        result = searcher.search(
            query="test query",
            strategy="fixed",
            top_k=3
        )
        
        assert result['metadata']['reranked'] is False
        assert 'rerank_ms' not in result['metadata']['timing']
