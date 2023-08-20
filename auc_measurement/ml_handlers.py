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

from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from sklearn.utils.multiclass import type_of_target
from typing import List, Tuple, Optional

from auc_measurement.config import Config
from auc_measurement.registries import MODEL_REGISTRY
from auc_measurement.scores import Scores


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


class ScoreHandler(ABC):
    """Encapsulates behavior of how to generate scores for different types of output variables.
    
    We have different logic for scoring binary vs multiclass data. (And probably binary one-vs-all
    still to come.) This class hierarchy factors those cases into separate classes. See:
      - BinaryScoreHandler
      - MulticlassScoreHandler
    """
    @abstractmethod
    def score(self, y_true: np.ndarray, y_predicted_soft: np.ndarray, y_predicted_hard: np.ndarray) -> Scores:
        """Compute all scores metrics for a given ground truth and sets of predictions.

        This runs the entire set of metrics that we track - per scores.Scores - on model predictions.
        Subclasses should handle logic distinguishing between binary and multiclass; the abstract base
        handles metrics that apply to both.

        Args:
            y_true (np.ndarray): The ground truth labels.
            y_predicted_soft (np.ndarray): "Soft" predictions - either probabilities (per predict_proba) or
                decision thresholds, per decision_function().
            y_predicted_hard (np.ndarray): "Hard" predictions - final predictions that have been cast to a
                discrete label value via the classifier's predict() function.

        Returns:
            Scores: Container of all metric values.
        """
        return Scores(
            auc=float(metrics.roc_auc_score(y_true=y_true,
                                            y_score=y_predicted_soft,
                                            average='weighted',
                                            multi_class='ovo')),
            f1=float(metrics.f1_score(y_true=y_true, y_pred=y_predicted_hard, average='weighted')),
            accuracy=float(metrics.accuracy_score(y_true=y_true, y_pred=y_predicted_hard))
        )

    @classmethod
    def score_handler_factory(cls, y_true: np.ndarray) -> 'ScoreHandler':
        if type_of_target(y_true) == 'multiclass':
            return MulticlassScoreHandler()
        elif type_of_target(y_true) == 'binary':
            return BinaryScoreHandler()
        else:
            raise ExperimentConfigurationException('Not sure how we got to scoring without realizing '
                                                   'that this data set is neither binary nor '
                                                   'multiclass, but here we are. *shrug*')


class BinaryScoreHandler(ScoreHandler):
    def score(self, y_true: np.ndarray, y_predicted_soft: np.ndarray, y_predicted_hard: np.ndarray) -> Scores:
        y_predicted_soft = y_predicted_soft[:, 1]
        result = super().score(y_true=y_true,
                               y_predicted_soft=y_predicted_soft,
                               y_predicted_hard=y_predicted_hard)
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_predicted_soft)
        result.roc_fpr.append(fpr)
        result.roc_tpr.append(tpr)
        result.roc_thresholds.append(thresholds)
        return result


class MulticlassScoreHandler(ScoreHandler):
    def score(self, y_true: np.ndarray, y_predicted_soft: np.ndarray, y_predicted_hard: np.ndarray) -> Scores:
        result = super().score(y_true=y_true,
                               y_predicted_soft=y_predicted_soft,
                               y_predicted_hard=y_predicted_hard)
        # We've trained in multiclass mode, but for ROC we need binary data. So we'll do
        # one-vs-all for each label.
        y_true = LabelBinarizer().fit(y_true).transform(y_true)  # type: ignore
        for one_v_all_label, pred in zip(y_true.T, y_predicted_soft.T):
            fpr, tpr, thresholds = metrics.roc_curve(y_true=one_v_all_label, y_score=pred)
            result.roc_fpr.append(fpr)
            result.roc_tpr.append(tpr)
            result.roc_thresholds.append(thresholds)
        return result


class MLExperimentEngine(object):
    """Container for the various handlers that are necessary to fully realize an experiment."""
    def __init__(self, config: Config, dataset: Bunch, dataset_base_name: str) -> None:
        self.config: Config = config
        self.dataset = dataset
        self.dataset_base_name: str = dataset_base_name
        self.split_handler: DataSplitHandler = self.data_split_handler_factory()
        self.y_type: str = type_of_target(dataset.target)
        if self.y_type not in ('binary', 'multiclass'):
            logging.warning(f'  Dataset {dataset_base_name} is neither binary nor multiclass; skipping.')
            raise ExperimentConfigurationException(f'Dataset {dataset_base_name} is neither binary nor multiclass')
        self.preprocessors: PreprocessorChain = []
        self.model: Optional[BaseEstimator] = None
        self.model_name: str = ''

    def get_row_count(self) -> int:
        return self.dataset['data'].shape[0]

    def dataset_is_large(self) -> bool:
        """Return true iff the current data set at least as large as the configured large_data_threshold."""
        return self.get_row_count() >= self.config.large_data_threshold

    def data_split_handler_factory(self) -> DataSplitHandler:
        """Get a data splitter specific to a given data set and configuration.

        Decides which splitter to employ and how many data splits (folds) it should generate,
        depending on the size of the dataset and the configuration.

        Args:
            config (Config): Experiment configuration.
            dataset (Bunch): Data set, including at least a 'data' field.

        Returns:
            DataSplitHandler: Appropriate DataSplitHandler object.
        """
        if self.dataset_is_large():
            logging.info(f'  Dataset has {self.get_row_count()} points; considered LARGE.')
            return DataSplitHandler(self.config.large_data.folds, random_state=self.config.random_seed)
        else:
            logging.info(f'  Dataset has {self.get_row_count()} points; considered SMALL.')
            return DataSplitHandler(self.config.small_data.folds, random_state=self.config.random_seed)

    def add_preprocessor(self, preprocessor_name: str, preprocessor: TransformerMixin):
        self.preprocessors.append((preprocessor_name, preprocessor))

    def remove_preprocessor(self, preprocessor_name: str):
        self.preprocessors = [x for x in self.preprocessors if x[0] != preprocessor_name]

    def maybe_add_calibrator(self, base_model: BaseEstimator) -> BaseEstimator:
        if hasattr(base_model, 'predict_proba'):
            # Model knows how to predict probabilities natively. Doesn't need calibration.
            logging.info(f"  Model {self.model_name} understands probabilities. Righteous. No calibration needed.")
            return base_model
        logging.info(f"  Model {self.model_name} doesn't know how to estimate probabilities natively. "
                     "Adding a probability calibration step.")
        if self.dataset_is_large():
            logging.info(f'  Dataset has {self.get_row_count()} points: using LARGE settings '
                         'for probability calibration.')
            return CalibratedClassifierCV(base_model,
                                          cv=self.config.large_data.calibration_folds,
                                          method=self.config.large_data.calibration_type)  # type: ignore
        else:
            logging.info(f'  Dataset has {self.get_row_count()} points: using SMALL settings '
                         'for probability calibration.')
            return CalibratedClassifierCV(base_model,
                                          cv=self.config.small_data.calibration_folds,
                                          method=self.config.small_data.calibration_type)  # type: ignore

    def set_model_by_name(self, model_name: str):
        try:
            model = MODEL_REGISTRY[model_name](**self.config.models_to_test[model_name],
                                               random_state=self.config.random_seed)
        except KeyError:
            raise ExperimentConfigurationException(f"You asked for a model named '{model_name}', but that name "
                                                   f"isn't registered in the known model names registry. See "
                                                   f"registries.py for known model names.")
        self.model_name = model_name
        calibrated_model = self.maybe_add_calibrator(model)
        self.model = Pipeline(self.preprocessors + [(f'Model_{model_name}', calibrated_model)])  # type: ignore

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
        else:
            raise ExperimentConfigurationException(f"Model {self.model_name} doesn't have a predict_proba() "
                                                   f"method. I don't know what to do with such a model.")
