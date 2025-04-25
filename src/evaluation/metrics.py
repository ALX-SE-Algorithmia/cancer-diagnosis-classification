"""
Metrics for evaluating rare genetic disorders classification models.
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels

    Returns:
        Accuracy score
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision_recall(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Tuple[float, float]:
    """
    Calculate precision and recall.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Tuple of (precision, recall)
    """
    if num_classes > 2:
        # Macro-averaged precision and recall for multi-class
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
    else:
        # Binary classification
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

    return precision, recall


def calculate_f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """
    Calculate F1 score.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        F1 score
    """
    if num_classes > 2:
        # Macro-averaged F1 for multi-class
        return f1_score(y_true, y_pred, average="macro")
    else:
        # Binary classification
        return f1_score(y_true, y_pred)


def calculate_auc(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    """
    Calculate AUC (Area Under the ROC Curve).

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        AUC score
    """
    try:
        if num_classes > 2:
            # One-vs-Rest AUC for multi-class
            y_true_binary = np.eye(num_classes)[y_true.astype(int)]
            y_pred_binary = np.eye(num_classes)[y_pred.astype(int)]
            return roc_auc_score(y_true_binary, y_pred_binary, average="macro", multi_class="ovr")
        else:
            # Binary classification
            return roc_auc_score(y_true, y_pred)
    except ValueError:
        # In case of only one class present in y_true
        return 0.5  # Default value for random classifier


def calculate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> plt.Figure:
    """
    Calculate and plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        Matplotlib figure with confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    
    # Set tick labels
    if num_classes <= 10:  # Only show class indices for a reasonable number of classes
        ax.set_xticks(np.arange(num_classes) + 0.5)
        ax.set_yticks(np.arange(num_classes) + 0.5)
        ax.set_xticklabels(np.arange(num_classes))
        ax.set_yticklabels(np.arange(num_classes))
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, y_scores: np.ndarray, num_classes: int
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores (probabilities)
        num_classes: Number of classes

    Returns:
        Matplotlib figure with ROC curve plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(
            fpr, 
            tpr, 
            lw=2, 
            label=f"ROC curve (area = {roc_auc:.2f})"
        )
    else:
        # Multi-class classification
        # One-hot encode the labels
        y_true_onehot = np.eye(num_classes)[y_true.astype(int)]
        
        # Plot ROC curve for each class
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(
                fpr, 
                tpr, 
                lw=2, 
                label=f"Class {i} (area = {roc_auc:.2f})"
            )
    
    # Plot random classifier
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, y_scores: np.ndarray, num_classes: int
) -> plt.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores (probabilities)
        num_classes: Number of classes

    Returns:
        Matplotlib figure with precision-recall curve plot
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        ax.plot(
            recall, 
            precision, 
            lw=2, 
            label=f"PR curve (area = {pr_auc:.2f})"
        )
    else:
        # Multi-class classification
        # One-hot encode the labels
        y_true_onehot = np.eye(num_classes)[y_true.astype(int)]
        
        # Plot PR curve for each class
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_scores[:, i])
            pr_auc = auc(recall, precision)
            
            ax.plot(
                recall, 
                precision, 
                lw=2, 
                label=f"Class {i} (area = {pr_auc:.2f})"
            )
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    
    return fig
