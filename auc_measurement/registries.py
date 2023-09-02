"""Registry objects for data loaders and models.

Each registry is a dict mapping from a string to a factory that
generates an appropriate object."""

from typing import Dict, Callable
from sklearn import (
    linear_model,
    svm,
    tree,
    ensemble,
    naive_bayes,
    discriminant_analysis,
    neighbors,
    gaussian_process,
    neural_network,
)
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    fetch_20newsgroups_vectorized,
    fetch_covtype,
    fetch_kddcup99,
    fetch_lfw_people,
    fetch_lfw_pairs,
    fetch_olivetti_faces,
    fetch_rcv1,
)

# This is mostly auto-populated from sklearn.utils.discovery.all_estimators('classifier')
# but we've filtered down some meta-classifiers.
MODEL_REGISTRY: Dict[str, Callable] = {
    'AdaBoostClassifier': ensemble.AdaBoostClassifier,
    'BaggingClassifier': ensemble.BaggingClassifier,
    'BernoulliNB': naive_bayes.BernoulliNB,
    'CategoricalNB': naive_bayes.CategoricalNB,  # Note: Requires non-negative feature data.
    'ComplementNB': naive_bayes.ComplementNB,
    'DecisionTreeClassifier': tree.DecisionTreeClassifier,
    'ExtraTreeClassifier': tree.ExtraTreeClassifier,
    'ExtraTreesClassifier': ensemble.ExtraTreesClassifier,
    'GaussianNB': naive_bayes.GaussianNB,
    'PassiveAggressiveClassifier': linear_model.PassiveAggressiveClassifier,
    'Perceptron': linear_model.Perceptron,
    'QuadraticDiscriminantAnalysis': discriminant_analysis.QuadraticDiscriminantAnalysis,
    'RadiusNeighborsClassifier': neighbors.RadiusNeighborsClassifier,
    'RandomForestClassifier': ensemble.RandomForestClassifier,
    'RidgeClassifier': linear_model.RidgeClassifier,
    'RidgeClassifierCV': linear_model.RidgeClassifierCV,
    'SGDClassifier': linear_model.SGDClassifier,
    'SVC': svm._classes.SVC,
    # Ensemble classifiers require specification of the sub-classifers. That's fine, but
    # omit for first draft.
    # 'StackingClassifier': ensemble.StackingClassifier,
    # 'VotingClassifier': ensemble.VotingClassifier,
    'GaussianProcessClassifier': gaussian_process.GaussianProcessClassifier,
    'GradientBoostingClassifier': ensemble.GradientBoostingClassifier,
    'HistGradientBoostingClassifier': ensemble.HistGradientBoostingClassifier,
    'KNeighborsClassifier': neighbors.KNeighborsClassifier,
    'LinearDiscriminantAnalysis': discriminant_analysis.LinearDiscriminantAnalysis,
    'LinearSVC': svm.LinearSVC,
    'LogisticRegression': linear_model.LogisticRegression,
    'LogisticRegressionCV': linear_model.LogisticRegressionCV,
    'MLPClassifier': neural_network.MLPClassifier,
    'MultinomialNB': naive_bayes.MultinomialNB,
    # NearestCentroid model doesn't define a soft classification method, so
    # not clear how to use it for thresholded performance metrics.
    # 'NearestCentroid': neighbors.NearestCentroid,
    'NuSVC': svm.NuSVC,
}

DATA_LOADER_REGISTRY: Dict[str, Callable] = {
    'breast_cancer': load_breast_cancer,
    'digits': load_digits,
    'iris': load_iris,
    'wine': load_wine,
    'news': fetch_20newsgroups_vectorized,
    'covtype': fetch_covtype,
    'kddcup99': fetch_kddcup99,
    'lfw_pairs': fetch_lfw_pairs,
    'lfw_people': fetch_lfw_people,
    'olivetti_faces': fetch_olivetti_faces,
    'rcv1': fetch_rcv1,
}
