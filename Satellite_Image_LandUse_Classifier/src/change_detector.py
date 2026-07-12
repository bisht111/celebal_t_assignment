import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def compute_roc(y_true, similarity_scores):

    fpr, tpr, thresholds = roc_curve(
        y_true,
        similarity_scores
    )

    roc_auc = auc(fpr, tpr)

    return fpr, tpr, thresholds, roc_auc


def find_best_threshold(fpr, tpr, thresholds):
    """
    Youden Index
    """

    scores = tpr - fpr

    idx = np.argmax(scores)

    return thresholds[idx]


def detect_change(similarity, threshold):

    return similarity < threshold