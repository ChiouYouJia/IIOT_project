import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)


def evaluate(y_true, y_pred, y_prob=None, average="weighted"):
    """Compute standard classification metrics."""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 1:
                results["roc_auc"] = roc_auc_score(y_true, y_prob)
            else:
                results["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
        except ValueError:
            results["roc_auc"] = None
    return results


def print_results(results, title="Results"):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f" Accuracy:  {results['accuracy']:.4f}")
    print(f" Precision: {results['precision']:.4f}")
    print(f" Recall:    {results['recall']:.4f}")
    print(f" F1-Score:  {results['f1']:.4f}")
    if "roc_auc" in results and results["roc_auc"] is not None:
        print(f" ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\n{results['report']}")
