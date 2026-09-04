import json
from typing import Dict
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    return metrics


def save_confusion_matrix(model, X_test, y_test, output_path: str = "confusion_matrix.png"):
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, cmap=plt.cm.Blues
    )
    plt.title("Confusion Matrix")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
