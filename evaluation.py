"""Evaluation utilities for hallucination detection."""

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_ranking(y_true, scores, method_name):
    """Evaluate a hallucination risk score.

    y_true uses 1 = hallucinated and 0 = correct.
    scores should be larger for higher hallucination risk.
    """
    return {
        "method": method_name,
        "AUROC": roc_auc_score(y_true, scores),
        "AUPRC": average_precision_score(y_true, scores),
    }


def summarize_methods(y_true, score_dict):
    """Evaluate multiple scoring methods and return a dataframe."""
    rows = [evaluate_ranking(y_true, scores, name) for name, scores in score_dict.items()]
    return pd.DataFrame(rows).sort_values("AUROC", ascending=False)
