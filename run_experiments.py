import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from grader_contracts.linear_models import RegressionInput, ClassificationInput
from linear_models import train_linear_regression, train_logistic_classifier, compare_classical_models

def metrics(y, probabilities, predictions):
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
    }

def main():
    root = Path(__file__).resolve().parent

    regression = pd.read_csv(root / "Student_Performance.csv")
    regression["Extracurricular Activities"] = (
        regression["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    )
    x = regression.drop(columns=["Performance Index"]).to_numpy(dtype=float)
    y = regression["Performance Index"].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    reg_result = train_linear_regression(
        RegressionInput(x_train, y_train, x_test, y_test, 0.05, 800)
    )
    test_mse = float(mean_squared_error(y_test, reg_result.predictions))

    german = pd.read_csv(root / "german.csv", sep=";")
    x = german.drop(columns=["Creditability"]).to_numpy(dtype=float)
    y = german["Creditability"].to_numpy(dtype=int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    cls_input = ClassificationInput(x_train, y_train, x_test, y_test, 0.1, 1000)
    own = train_logistic_classifier(cls_input)
    own_metrics = metrics(y_test, own.probabilities, own.predictions)
    baselines = compare_classical_models(cls_input)

    def to_dict(value):
        return {key: float(getattr(value, key))
                for key in ("roc_auc", "accuracy", "precision", "recall", "f1")}

    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_metrics,
            "logistic_regression": to_dict(baselines.logistic_regression),
            "decision_tree": to_dict(baselines.decision_tree),
            "knn": to_dict(baselines.knn),
        },
    }
    (root / "experiment_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
