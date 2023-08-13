"""Handlers and corresponding factories to abstract away a lot of ML case logic.

The issue is that we have the Cartesian product of a number of factors that we need
to deal with:
  - Choice of model (depends on config)
  - Number of data folds (depends on data set and config)
  - Preprocessing steps (depends on data set X and y)
  - How to do soft predictions (depends on model)
  - How to handle scoring (depends on data set y type)

We could inline all of those decisions, but that leads to somewhat gnarly code with
a maze of twisty conditionals.  Instead, we can factor the decision logic out into a set
of factory functions that encode the necessary case logic and that produce handlers that
encapsulate the case-specific behaviors.
"""

import logging
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.utils.multiclass import type_of_target
from typing import List, Tuple, Optional

from auc_measurement.config import Config
from auc_measurement.registries import MODEL_REGISTRY


class ExperimentConfigurationException(Exception):
    """Indicates that some configuration is inconsistent, impossible, or model incopatible."""
    pass


# Type definition for a sequence of preprocessors.
PreprocessorChain = List[Tuple[str, TransformerMixin]]


class DataSplitHandler(object):
    """Encapsulates the behavior of a cross-validation splitter.

    For the moment, there is only one type of DataSplitHandler and it implements only
    StratifiedShuffleSplit.
    """
    def __init__(self, n_splits: int, random_state: int) -> None:
        self._n_splits = n_splits
        self._cv = StratifiedShuffleSplit(n_splits=self._n_splits, random_state=random_state)

    def split_data(self, X, y):
        return self._cv.split(X, y)


def data_split_handler_factory(config: Config, dataset: Bunch) -> DataSplitHandler:
    """Get a data splitter specific to a given data set and configuration.

    Decides which splitter to employ and how many data splits (folds) it should generate,
    depending on the size of the dataset and the configuration.

    Args:
        config (Config): Experiment configuration.
        dataset (Bunch): Data set, including at least a 'data' field.

    Returns:
        DataSplitHandler: Appropriate DataSplitHandler object.
    """
    rows = dataset['data'].shape[0]
    if rows < config.large_data_threshold:
        logging.info(f'  Dataset has {rows} points; considered SMALL.')
        return DataSplitHandler(config.small_data.folds, random_state=config.random_seed)
    else:
        logging.info(f'  Dataset has {rows} points; considered LARGE.')
        return DataSplitHandler(config.large_data.folds, random_state=config.random_seed)


class MLExperimentHandlerSet(object):
    """Container for the various handlers that are necessary to fully realize an experiment."""
    def __init__(self, config: Config, dataset: Bunch, dataset_base_name: str) -> None:
        self.config: Config = config
        self.dataset_base_name: str = dataset_base_name
        self.split_handler: DataSplitHandler = data_split_handler_factory(config=config, dataset=dataset)
        self.y_type: str = type_of_target(dataset.target)
        if self.y_type not in ('binary', 'multiclass'):
            logging.warning(f'Dataset {dataset_base_name} is neither binary nor multiclass; skipping.')
            raise ExperimentConfigurationException(f'Dataset {dataset_base_name} is neither binary nor multiclass')
        self.preprocessors: PreprocessorChain = []
        self.model: Optional[BaseEstimator] = None
        self.model_name: str = ''

    def add_preprocessor(self, preprocessor_name: str, preprocessor: TransformerMixin):
        self.preprocessors.append((preprocessor_name, preprocessor))

    def remove_preprocessor(self, preprocessor_name: str):
        self.preprocessors = [x for x in self.preprocessors if x[0] != preprocessor_name]

    def set_model_by_name(self, model_name: str):
        try:
            model = MODEL_REGISTRY[model_name](**self.config.models_to_test[model_name],
                                               random_state=self.config.random_seed)
        except KeyError:
            raise ExperimentConfigurationException(f"You asked for a model named '{model_name}', but that name "
                                                   f"isn't registered in the known model names registry. See "
                                                   f"registries.py for known model names.")
        self.model = Pipeline(self.preprocessors + [(f'Model_{model_name}', model)])  # type: ignore
        self.model_name = model_name

    def fit_model(self, X: np.ndarray, y: np.ndarray):
        if self.model is None:
            raise ValueError('You need to set a model in the MLExperimentHandlerSet before calling fit_model')
        # Annoyingly, there is no single base class in sklearn that declares the fit() method. WTF people.
        self.model.fit(X, y)  # type: ignore

    def predict_soft(self, X: np.ndarray) -> np.ndarray:
        """Figure out whether to use decision_function or predict_proba.

        Args:
            X (np.ndarray): Data feature set.

        Returns:
            np.ndarray: Predicted confidence/margin/soft values.
        """
        if self.model is None:
            raise ValueError('You need to set a model in the MLExperimentHandlerSet before calling predict_soft.')
        if hasattr(self.model, 'predict_proba'):
            logging.debug(f'Model {self.model_name} supports predict_proba.')
            return self.model.predict_proba(X)  # type: ignore
        elif hasattr(self.model, 'decision_function'):
            logging.debug(f"Model {self.model_name} doesn't support predict_proba.")
            return self.model.decision_function(X)  # type: ignore
        else:
            raise ExperimentConfigurationException(f"Model {self.model_name} has neither a predict_proba() "
                                                   f"nor a decision_function method. I don't know what to "
                                                   f"do with such a model.")
