"""Model evaluation utilities."""

from .metrics import compute_metrics
from .visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importances
)
from .threshold import (
    tune_threshold_f1,
    tune_threshold_precision_constrained_recall,
    tune_threshold_top_k,
    tune_threshold_cost_sensitive,
    evaluate_threshold
)

__all__ = [
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_precision_recall_curve',
    'plot_feature_importances',
    'tune_threshold_f1',
    'tune_threshold_precision_constrained_recall',
    'tune_threshold_top_k',
    'tune_threshold_cost_sensitive',
    'evaluate_threshold'
]

