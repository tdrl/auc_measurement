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
from sklearn.base import TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import Bunch
from sklearn.utils.multiclass import type_of_target
from typing import List, Tuple

from auc_measurement.config import Config


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
        self.dataset_base_name: str = dataset_base_name
        self.split_handler: DataSplitHandler = data_split_handler_factory(config=config, dataset=dataset)
        self.y_type = type_of_target(dataset.target)
        if self.y_type not in ('binary', 'multiclass'):
            logging.warning(f'Dataset {dataset_base_name} is neither binary nor multiclass; skipping.')
            raise ExperimentConfigurationException(f'Dataset {dataset_base_name} is neither binary nor multiclass')
        self.preprocessors: PreprocessorChain = []

    def add_preprocessor(self, preprocessor_name: str, preprocessor: TransformerMixin):
        self.preprocessors.append((preprocessor_name, preprocessor))

    def remove_preprocessor(self, preprocessor_name: str):
        self.preprocessors = [x for x in self.preprocessors if x[0] != preprocessor_name]
