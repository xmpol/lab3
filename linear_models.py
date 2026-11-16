from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)

def _as_2d(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Признаки должны быть двумерным массивом")
    return array

def _sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))

def _fit_logistic(x_train, y_train, learning_rate, epochs):
    n_samples, n_features = x_train.shape
    weights = np.zeros(n_features, dtype=float)
    bias = 0.0
    losses = []
    y = np.asarray(y_train, dtype=float).reshape(-1)
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("Целевая переменная логистической регрессии должна быть 0/1")

    for _ in range(int(epochs)):
        probabilities = _sigmoid(x_train @ weights + bias)
        eps = 1e-12
        loss = -np.mean(y * np.log(probabilities + eps)
                        + (1.0 - y) * np.log(1.0 - probabilities + eps))
        losses.append(float(loss))
        error = probabilities - y
        weights -= learning_rate * (x_train.T @ error) / n_samples
        bias -= learning_rate * float(np.mean(error))
    return weights, bias, losses

def _classification_metrics(y_true, probabilities, predictions):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, probabilities)),
        accuracy=float(accuracy_score(y_true, predictions)),
        precision=float(precision_score(y_true, predictions, zero_division=0)),
        recall=float(recall_score(y_true, predictions, zero_division=0)),
        f1=float(f1_score(y_true, predictions, zero_division=0)),
    )

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    if x_train.shape[0] != y_train.size:
        raise ValueError("Число объектов и целевых значений train не совпадает")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std

    weights_scaled = np.zeros(x_train_scaled.shape[1], dtype=float)
    bias_scaled = 0.0
    losses = []
    n_samples = x_train_scaled.shape[0]
    for _ in range(int(data.epochs)):
        predictions_scaled = x_train_scaled @ weights_scaled + bias_scaled
        errors = predictions_scaled - y_scaled
        losses.append(float(np.mean((errors * y_std) ** 2)))
        weights_scaled -= data.learning_rate * (2.0 / n_samples) * (
            x_train_scaled.T @ errors
        )
        bias_scaled -= data.learning_rate * 2.0 * float(np.mean(errors))

    weights = y_std * weights_scaled / scaler.scale_
    bias = y_mean + y_std * bias_scaled - float(np.sum(weights * scaler.mean_))
    predictions = x_test @ weights + bias
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=losses,
        predictions=np.asarray(predictions, dtype=float),
    )

def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    weights_scaled, bias_scaled, losses = _fit_logistic(
        x_train_scaled, y_train, data.learning_rate, data.epochs
    )
    probabilities = _sigmoid(x_test_scaled @ weights_scaled + bias_scaled)

    weights = weights_scaled / scaler.scale_
    bias = float(bias_scaled - np.sum(weights * scaler.mean_))
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=losses,
        probabilities=probabilities,
        predictions=predictions,
    )

def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    from sklearn.linear_model import LogisticRegression

    x_train = _as_2d(data.train_features)
    x_test = _as_2d(data.test_features)
    y_train = np.asarray(data.train_target).reshape(-1).astype(int)
    y_test = np.asarray(data.test_target).reshape(-1).astype(int)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logistic = LogisticRegression(max_iter=2000, random_state=42)
    logistic.fit(x_train_scaled, y_train)
    logistic_prob = logistic.predict_proba(x_test_scaled)[:, 1]
    logistic_pred = (logistic_prob >= 0.5).astype(int)

    tree = DecisionTreeClassifier(random_state=42, max_depth=5)
    tree.fit(x_train, y_train)
    tree_prob = tree.predict_proba(x_test)[:, 1]
    tree_pred = tree.predict(x_test)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(x_train_scaled, y_train)
    knn_prob = knn.predict_proba(x_test_scaled)[:, 1]
    knn_pred = knn.predict(x_test_scaled)

    return BaselineComparisonResult(
        logistic_regression=_classification_metrics(y_test, logistic_prob, logistic_pred),
        decision_tree=_classification_metrics(y_test, tree_prob, tree_pred),
        knn=_classification_metrics(y_test, knn_prob, knn_pred),
    )
