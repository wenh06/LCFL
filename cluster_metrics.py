from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

__all__ = ["compute_clustering_metrics"]


def compute_clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute clustering metrics, including ARI, NMI, and clustering accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        True cluster labels of the data points.
    y_pred : np.ndarray
        Predicted cluster labels of the data points.

    Returns
    -------
    metrics : Dict[str, float]
        A dictionary containing the computed clustering metrics.

    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    metrics = {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "ACC": clustering_accuracy(y_true, y_pred),
    }
    metrics = {k: v.item() if isinstance(v, np.generic) else v for k, v in metrics.items()}
    return metrics


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute clustering accuracy with optimal label matching using Hungarian algorithm.

    Parameters
    ----------
    y_true : np.ndarray
        True cluster labels of the data points.
    y_pred : np.ndarray
        Predicted cluster labels of the data points.

    Returns
    -------
    acc : float
        The clustering accuracy.
    """
    label_true = LabelEncoder().fit_transform(y_true)
    label_pred = LabelEncoder().fit_transform(y_pred)

    D = max(label_true.max(), label_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(label_pred.size):
        w[label_pred[i], label_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = w[row_ind, col_ind].sum() / y_pred.size
    return acc.item()
