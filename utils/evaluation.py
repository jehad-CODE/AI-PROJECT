from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Evaluate the model using accuracy, precision, recall, F1-score, and optionally ROC-AUC.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        y_prob (array): Predicted probabilities (optional for ROC-AUC).

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }
    
    # Include ROC-AUC if probabilities are provided
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        
    return metrics
