import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from data import CLASSES


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(len(CLASSES)))
    is_attack = y_true != 0
    flagged = y_pred != 0

    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "detection_rate": (flagged & is_attack).sum() / is_attack.sum(),
        "fpr": (flagged & ~is_attack).sum() / (~is_attack).sum(),
        "alert_precision": (flagged & is_attack).sum() / max(flagged.sum(), 1),
    }
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for name, score in zip(CLASSES, per_class):
        scores[f"f1_{name}"] = score
    return scores
