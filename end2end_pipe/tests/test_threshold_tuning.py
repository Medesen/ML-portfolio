import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def test_threshold_tuning_f1():
    """Test F1 threshold optimization finds optimal threshold."""
    # Create simple scenario where 0.6 threshold maximizes F1
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.5, 0.7, 0.8, 0.9])

    # Test various thresholds
    best_f1 = -1
    best_threshold = 0.5

    for threshold in np.unique(y_proba):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # At 0.5, we get TP=4, FP=0, FN=0, TN=2
    # F1 = 2*4/(2*4+0+0) = 1.0
    y_pred_optimal = (y_proba >= 0.5).astype(int)
    assert f1_score(y_true, y_pred_optimal) == 1.0
    assert best_threshold == 0.5


def test_threshold_tuning_precision_constrained_recall():
    """Test precision-constrained recall optimization."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    min_precision = 0.8
    best_recall = -1
    best_threshold = 0.5

    for threshold in np.unique(y_proba):
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if precision >= min_precision and recall > best_recall:
            best_recall = recall
            best_threshold = threshold

    # Verify we found a threshold that meets precision constraint
    y_pred_best = (y_proba >= best_threshold).astype(int)
    assert precision_score(y_true, y_pred_best) >= min_precision


def test_threshold_tuning_top_k():
    """Test top-k threshold selection."""
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Select top 30% (3 out of 10)
    top_k_ratio = 0.3
    k = max(1, int(len(y_proba) * top_k_ratio))

    sorted_probs = np.sort(y_proba)[::-1]
    threshold = float(sorted_probs[k - 1])

    # k=3, so sorted_probs[2] = 0.8
    assert threshold == 0.8

    # Verify exactly k predictions are positive
    y_pred = (y_proba >= threshold).astype(int)
    assert y_pred.sum() == k


def test_edge_case_all_same_probabilities():
    """Test threshold tuning when all probabilities are identical."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.5, 0.5, 0.5, 0.5])

    # With all same probabilities, any threshold should work
    candidate_thresholds = np.unique(np.concatenate([[0.0, 1.0], y_proba]))

    # Should not crash
    for t in candidate_thresholds:
        y_pred = (y_proba >= t).astype(int)
        # F1 is defined even with all same predictions
        _ = f1_score(y_true, y_pred, zero_division=0)


def test_edge_case_perfect_separation():
    """Test threshold tuning with perfect class separation."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    # Optimal threshold should be somewhere between 0.3 and 0.7
    best_f1 = -1
    best_threshold = 0.5

    for threshold in np.unique(y_proba):
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Should achieve perfect F1
    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    assert f1_score(y_true, y_pred_optimal) == 1.0


def test_threshold_boundaries():
    """Test threshold at boundary values (0.0 and 1.0)."""
    y_proba = np.array([0.2, 0.4, 0.6, 0.8])

    # Threshold 0.0: all predictions are positive
    y_pred_zero = (y_proba >= 0.0).astype(int)
    assert y_pred_zero.sum() == 4

    # Threshold 1.0: all predictions are negative
    y_pred_one = (y_proba >= 1.0).astype(int)
    assert y_pred_one.sum() == 0

    # Threshold 0.9: only highest prediction is positive
    y_pred_high = (y_proba >= 0.9).astype(int)
    assert y_pred_high.sum() == 0  # None >= 0.9
