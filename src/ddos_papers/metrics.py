from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, precision_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    safe = lambda n, d: float(n / d) if d else 0.0
    return {
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "false_positive_rate": safe(fp, fp + tn),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "npv": safe(tn, tn + fn),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def regression_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    error = y_true - scores
    mse = mean_squared_error(y_true, scores)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, scores)),
        "sse": float(np.sum(error ** 2)),
    }
