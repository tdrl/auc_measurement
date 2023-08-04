"""Registry objects for data loaders and models.

Each registry is a dict mapping from a string to a factory that
generates an appropriate object."""

from typing import Dict, Callable
from sklearn import (
    linear_model,
    svm,
    tree,
)
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)

MODEL_REGISTRY: Dict[str, Callable] = {
    'dtree': tree.DecisionTreeClassifier,
    'logistic': linear_model.LogisticRegression,
    'ridge_classifier': linear_model.RidgeClassifier,
    'svm': svm.SVC,
}

DATA_LOADER_REGISTRY: Dict[str, Callable] = {
    'breast_cancer': load_breast_cancer,
    'digits': load_digits,
    'iris': load_iris,
    'wine': load_wine,
}
