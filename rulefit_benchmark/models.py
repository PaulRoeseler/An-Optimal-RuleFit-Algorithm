from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR


def linear_regression_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


def logistic_regression_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = LogisticRegression(random_state=0, solver="lbfgs", max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def svm_classifier_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = SVC(random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def svm_regression_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = SVR()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


def knn_classifier_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


def knn_regression_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


def naive_bayes_pipeline(X_train, X_test, y_train, y_test) -> float:
    model = GaussianNB()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


CLASSIFICATION_PIPELINES: Dict[str, Callable[..., float]] = {
    "logistic_regression": logistic_regression_pipeline,
    "svm": svm_classifier_pipeline,
    "naive_bayes": naive_bayes_pipeline,
    "knn": knn_classifier_pipeline,
}
REGRESSION_PIPELINES: Dict[str, Callable[..., float]] = {
    "linear_regression": linear_regression_pipeline,
    "svm": svm_regression_pipeline,
    "knn": knn_regression_pipeline,
}


def get_pipeline(problem_type: str, name: str) -> Callable[..., float]:
    registry = CLASSIFICATION_PIPELINES if problem_type == "classification" else REGRESSION_PIPELINES
    if name not in registry:
        raise KeyError(f"Model '{name}' not available for {problem_type}. Available: {sorted(registry.keys())}")
    return registry[name]

