import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, balanced_accuracy_score


def expected_calibration_error(labels, y_pred_confidence, num_bins=15, adaptive=True):
    if adaptive:
        # Adaptive binning: Use quantiles to create bins
        bins = np.quantile(y_pred_confidence, np.linspace(0, 1, num_bins + 1))
    else:
        bins = np.linspace(0, 1, num_bins + 1) # Bin edges
    bin_counts = np.zeros(num_bins) # Number of predictions in each bin
    bin_correct = np.zeros(num_bins) # Number of correct predictions in each bin
    bin_confidence = np.zeros(num_bins) # Mean confidence in each bin

    for label, pred_conf in zip(labels, y_pred_confidence):
        bin_idx = np.digitize(pred_conf, bins, right=True) - 1
        if bin_idx >= num_bins: # Account for edge case
            bin_idx = num_bins - 1
        bin_counts[bin_idx] += 1
        bin_confidence[bin_idx] += pred_conf
        bin_correct[bin_idx] += label

    bin_accuracy = np.nan_to_num(bin_correct / bin_counts)
    bin_confidence = np.nan_to_num(bin_confidence / bin_counts)

    ece = np.sum(bin_counts * np.abs(bin_accuracy - bin_confidence)) / len(labels)
    return ece

def root_brier_score(labels, y_pred):
    brier_score = brier_score_loss(labels, y_pred)
    root_brier = np.sqrt(brier_score)
    return root_brier

def calculate_metrics(y_true, y_pred_probs, threshold=0.5, is_multilabel=False):
    """
    Calculate performance metrics for binary or multi-label classification.

    Args:
        y_true: Ground truth labels, shape (N,) for binary or (N, num_labels) for multi-label
        y_pred_probs: Predicted probabilities (softmax or sigmoid outputs), shape (N,) for binary or (N, num_labels) for multi-label
        threshold: Probability threshold to convert probabilities to binary labels (default 0.5)
        is_multilabel: Whether the task is multi-label classification (default False)

    Returns:
        Dictionary of performance metrics
    """
    # For binary classification
    if not is_multilabel:
        # For binary classification, y_pred_probs should be the probabilities for class 1 (positive class)
        y_pred = (y_pred_probs >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        bal_accuracy = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Confusion matrix to calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (fp + tn)  # True Negative / (False Positive + True Negative)

        auc = roc_auc_score(y_true, y_pred_probs)  # AUC for binary classification



        return {
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'auc': auc,
            'f1_score': f1,
            'TPr': tp/len(y_true),
            'TNr': tn/len(y_true),
            'FPr': fp/len(y_true),
            'FNr': fn/len(y_true),
        }

    # For multi-label classification
    else:
        # For multi-label classification, threshold the probabilities for each label/class
        y_pred = (y_pred_probs >= threshold).astype(int)

        # Calculate accuracy (average of per-label accuracy)
        bal_accuracy = balanced_accuracy_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)

        # Precision, recall, and F1 score averaged across all labels (macro-average)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # AUC for multi-label classification (average over all classes)
        auc = roc_auc_score(y_true, y_pred_probs, average='macro', multi_class='ovr')

        return {
            'accuracy': accuracy,
            'bal_accuracy': bal_accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'f1_score': f1,
        }

def negative_predictive_value(y_true, y_pred):
    """
    Calculate the negative predictive value (NPV) for binary classification.

    Parameters:
    y_true (np.array): The true labels.
    y_pred (np.array): The predicted labels.

    Returns:
    npv (float): The negative predictive value.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Avoid division by zero
    
    return npv

def calculate_calibration_metrics(true_labels, pos_confidences, num_bins=15):
 
    ece = expected_calibration_error(true_labels, pos_confidences, num_bins, adaptive=False)
    ace = expected_calibration_error(true_labels, pos_confidences, num_bins, adaptive=True)  

    return {
        'ece': ece,
        'adaece': ace,
        'rbs': root_brier_score(true_labels, pos_confidences)
    }



