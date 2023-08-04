"""Registry objects for data loaders and models.

Each registry is a dict mapping from a string to a factory that
generates an appropriate object."""

from typing import Dict, Callable
from sklearn import (
    svm,
    tree,
    linear_model,
)
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_diabetes,
)

MODEL_REGISTRY: Dict[str, Callable] = {
    'svm': svm.SVC,
    'dtree': tree.DecisionTreeClassifier,
    'logistic': linear_model.LogisticRegression,
    'ridge_classifier': linear_model.RidgeClassifier,
}

DATA_LOADER_REGISTRY: Dict[str, Callable] = {
    'iris': load_iris,
    'breast_cancer': load_breast_cancer,
    'diabetes': load_diabetes,
}