"""Unit tests for ml_handlers.py."""

from unittest import TestCase, main, skip
from pathlib import Path
from tempfile import mkdtemp
from os import chdir
from scipy import sparse
from shutil import rmtree
from sklearn import tree, svm, naive_bayes
from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize, MaxAbsScaler, MinMaxScaler, StandardScaler
from sklearn.utils import Bunch
import numpy as np
from numpy.testing import assert_array_almost_equal
from dataclasses import dataclass, field

from auc_measurement.config import Config
import auc_measurement.ml_handlers as target
import auc_measurement.registries as reg


@dataclass
class DatasetDefinition(object):
    """A convenience wrapper to hold common information about various test datasets."""
    name: str
    n_features: int
    n_rows: int
    n_classes: int
    X: np.ndarray
    y: np.ndarray
    dataset: Bunch = field(init=False)

    def __post_init__(self):
        self.dataset = Bunch(data=self.X, target=self.y)


class TestMlHandlers(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._origin = Path.cwd()
        self._temp_dir = Path(mkdtemp())
        chdir(self._temp_dir)
        self._default_config = Config(experiments_output_dir='.',
                                      # Populate with all known models, so we don't have to keep
                                      # this manually in sync with the registry.
                                      models_to_test={m_name: {} for m_name in reg.MODEL_REGISTRY.keys()},
                                      large_data_threshold=200)
        # Dataset sizes chosen to that (a) they're both SMALL, and (b) there're enough samples so that
        # each class has a reasonable number of exemplars. (Specifically, we have at least 5 examples of
        # each class after CV splitting).
        X_bin, y_bin = make_classification(n_samples=40, n_features=20, n_classes=2)
        X_sparse = sparse.rand(m=1000, n=100, density=0.01, format='coo')
        y_sparse = np.random.randint(0, 2, size=(100,))
        multi_n_classes = 7
        X_multi, y_multi = make_classification(n_samples=100,
                                               n_classes=multi_n_classes,
                                               n_features=11,
                                               n_informative=9)
        self._default_nonnegative_rows = 40
        self._default_nonnegative_features = 10
        X_nonnegative = np.random.random_sample(size=(self._default_nonnegative_rows,
                                                      self._default_nonnegative_features))
        y_nonnegative = np.random.randint(0, 2, size=(self._default_nonnegative_rows,))
        self.binary_data = DatasetDefinition(name='binary',
                                             n_rows=X_bin.shape[0],
                                             n_features=X_bin.shape[1],
                                             n_classes=2,
                                             X=X_bin,
                                             y=y_bin)
        self.sparse_data = DatasetDefinition(name='sparse',
                                             n_rows=X_sparse.shape[0],
                                             n_features=X_sparse.shape[1],
                                             n_classes=2,
                                             X=X_sparse,  # type: ignore
                                             y=y_sparse)
        self.multiclass_data = DatasetDefinition(name='multiclass',
                                                 n_rows=X_multi.shape[0],
                                                 n_features=X_multi.shape[1],
                                                 n_classes=multi_n_classes,
                                                 X=X_multi,
                                                 y=y_multi)
        self.nonnegative_data = DatasetDefinition(name='nonnegative',
                                                  n_rows=X_nonnegative.shape[0],
                                                  n_features=X_nonnegative.shape[1],
                                                  n_classes=2,
                                                  X=X_nonnegative,
                                                  y=y_nonnegative)
        self.all_datasets = {
            'binary': self.binary_data,
            'sparse': self.sparse_data,
            'multiclass': self.multiclass_data,
            'nonnegative': self.nonnegative_data,
        }

    def tearDown(self) -> None:
        chdir(self._origin)
        rmtree(self._temp_dir)
        return super().tearDown()

    def test_data_split_handler_recognizes_small_data(self):
        config = Config(experiments_output_dir='.', large_data_threshold=37)
        config.small_data.folds = 5
        config.large_data.folds = 2
        X, y = make_classification(n_samples=20)
        dataset = Bunch(data=X, target=y)
        experiment = target.MLExperimentEngine(config=config, dataset=dataset, dataset_base_name='testData')
        handler = experiment.data_split_handler_factory()
        self.assertEqual(handler._n_splits, 5)
        splits = handler.split_data(dataset.data, dataset.target)
        self.assertEqual(len(list(splits)), 5)

    def test_data_split_handler_recognizes_large_data(self):
        config = Config(experiments_output_dir='.', large_data_threshold=37)
        config.small_data.folds = 5
        config.large_data.folds = 2
        X, y = make_classification(n_samples=50)
        dataset = Bunch(data=X, target=y)
        experiment = target.MLExperimentEngine(config=config, dataset=dataset, dataset_base_name='testData')
        handler = experiment.data_split_handler_factory()
        self.assertEqual(handler._n_splits, 2)
        splits = handler.split_data(dataset.data, dataset.target)
        self.assertEqual(len(list(splits)), 2)

    def test_add_remove_preprocessors(self):
        handlers = target.MLExperimentEngine(config=self._default_config,
                                             dataset=self.binary_data.dataset,
                                             dataset_base_name='testData')
        self.assertEqual(len(handlers.preprocessors), 0)
        handlers.add_preprocessor('foo', None)  # type: ignore
        self.assertEqual(len(handlers.preprocessors), 1)
        handlers.add_preprocessor('bar', None)  # type: ignore
        self.assertEqual(len(handlers.preprocessors), 2)
        handlers.add_preprocessor('baz', None)  # type: ignore
        self.assertEqual(len(handlers.preprocessors), 3)
        handlers.remove_preprocessor('bar')
        self.assertEqual(len(handlers.preprocessors), 2)
        self.assertEqual([x[0] for x in handlers.preprocessors], ['foo', 'baz'])

    def test_set_model_exception_if_unknown_model_name(self):
        handlers = target.MLExperimentEngine(config=self._default_config,
                                             dataset=self.binary_data.dataset,
                                             dataset_base_name='testData')
        with self.assertRaisesRegex(target.ExperimentConfigurationException, r'blurfle'):
            handlers.set_model_by_name('blurfle')

    def test_predict_soft_error_if_model_unset(self):
        handlers = target.MLExperimentEngine(config=self._default_config,
                                             dataset=self.binary_data.dataset,
                                             dataset_base_name='testData')
        with self.assertRaises(ValueError):
            handlers.predict_soft(self.binary_data.X)

    def test_predict_all_classifiers_binary(self):
        engine = target.MLExperimentEngine(config=self._default_config,
                                           dataset=self.binary_data.dataset,
                                           dataset_base_name='testData')
        for model_name in target.MODEL_REGISTRY:
            engine.set_model_by_name(model_name)
            engine.setup_model_pre_post_processors(self.binary_data.dataset)
            engine.fit_model(self.binary_data.X, self.binary_data.y)
            yhat = engine.predict_soft(self.binary_data.X)
            self.assertEqual(yhat.shape, (self.binary_data.n_rows, 2), f'Failed on model = {model_name}')
            assert_array_almost_equal(yhat.sum(axis=1), np.ones((self.binary_data.n_rows,)),
                                      err_msg=f'Failed on model = {model_name}')

    def test_predict_all_classifiers_multiclass(self):
        engine = target.MLExperimentEngine(config=self._default_config,
                                           dataset=self.multiclass_data.dataset,
                                           dataset_base_name='testData')
        for model_name in target.MODEL_REGISTRY:
            engine.set_model_by_name(model_name)
            engine.setup_model_pre_post_processors(self.multiclass_data.dataset)
            engine.fit_model(self.multiclass_data.X, self.multiclass_data.y)
            yhat = engine.predict_soft(self.multiclass_data.X)
            self.assertEqual(yhat.shape, (self.multiclass_data.n_rows, self.multiclass_data.n_classes),
                             f'Failed on model = {model_name}')
            assert_array_almost_equal(yhat.sum(axis=1), np.ones((self.multiclass_data.n_rows,)),
                                      err_msg=f'Failed on model = {model_name}')

    def test_scorer_factory_bin(self):
        bin_y_data = np.concatenate([np.zeros((37,)), np.ones((28,))])
        scorer = target.ScoreHandler.score_handler_factory(bin_y_data)
        self.assertIsInstance(scorer, target.BinaryScoreHandler)

    def test_scorer_factory_multiclass(self):
        multi_y_data = np.repeat([0, 1, 2, 3, 4, 5], 8)  # Six classes
        scorer = target.ScoreHandler.score_handler_factory(multi_y_data)
        self.assertIsInstance(scorer, target.MulticlassScoreHandler)

    def test_score_factory_dispatch(self):
        self.assertIsInstance(target.ScoreHandler.score_handler_factory(self.binary_data.y),
                              target.BinaryScoreHandler)
        self.assertIsInstance(target.ScoreHandler.score_handler_factory(self.multiclass_data.y),
                              target.MulticlassScoreHandler)

    def test_score_binary_proba(self):
        scorer = target.BinaryScoreHandler()
        dtree = tree.DecisionTreeClassifier()
        X, y_true = self.binary_data.X, self.binary_data.y
        dtree.fit(X, y_true)
        y_predicted_soft = dtree.predict_proba(X)
        y_predicted_hard = dtree.predict(X)
        scores = scorer.score(y_true, y_predicted_soft, y_predicted_hard)  # type: ignore
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), 1,
                             msg=f'Failed length of {name} list = 1')
            self.assertGreater(roc_widget[0].shape[0], 2,
                               msg=f'Failed length of {name} vector is at least 2')
            self.assertLessEqual(roc_widget[0].shape[0], self.binary_data.n_rows + 1,
                                 msg=f'Failed {name} vector no longer than data set')

    def test_score_binary_decision_predictor(self):
        scorer = target.BinaryScoreHandler()
        svc = svm.SVC()
        X, y_true = self.binary_data.X, self.binary_data.y
        svc.fit(X, y_true)
        y_predicted_hard = svc.predict(X)
        # TODO(hlane): This is a mess. decision_function returns an (n,) vector, while predict_proba
        # returns a (n, k) matrix (for a k-class prediction). For the moment, gang it up here.
        y_predicted_soft = np.stack((np.zeros_like(y_predicted_hard), svc.decision_function(X)), axis=1)
        scores = scorer.score(y_true, y_predicted_soft, y_predicted_hard)
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), 1,
                             msg=f'Failed length of {name} list = 1')
            self.assertGreater(roc_widget[0].shape[0], 2,
                               msg=f'Failed length of {name} vector is at least 2')
            self.assertLessEqual(roc_widget[0].shape[0], self.binary_data.n_rows + 1,
                                 msg=f'Failed {name} vector no longer than data set')

    def test_score_binary_random_data_proba_model(self):
        scorer = target.BinaryScoreHandler()
        y_true = np.random.random_integers(0, 1, (self.binary_data.n_rows,))
        y_predicted_soft = np.random.random_sample((self.binary_data.n_rows, 2))
        y_predicted_soft = normalize(y_predicted_soft, norm='l1', axis=1)  # Ensure rows sum to 1.
        y_predicted_hard = np.random.randint(0, 1, (self.binary_data.n_rows))
        scores = scorer.score(y_true, y_predicted_soft, y_predicted_hard)  # type: ignore
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), 1,
                             msg=f'Failed length of {name} list = 1')
            self.assertGreater(roc_widget[0].shape[0], 2,
                               msg=f'Failed length of {name} vector is at least 2')
            self.assertLessEqual(roc_widget[0].shape[0], self.binary_data.n_rows + 1,
                                 msg=f'Failed {name} vector no longer than data set')

    def test_score_binary_random_data_decision_model(self):
        scorer = target.BinaryScoreHandler()
        y_true = np.random.random_integers(0, 1, (self.binary_data.n_rows,))
        y_predicted_soft = np.random.exponential(scale=5.0, size=(self.binary_data.n_rows, 2)) - 0.5
        y_predicted_hard = np.random.randint(0, 1, (self.binary_data.n_rows))
        scores = scorer.score(y_true, y_predicted_soft, y_predicted_hard)
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), 1,
                             msg=f'Failed length of {name} list = 1')
            self.assertGreater(roc_widget[0].shape[0], 2,
                               msg=f'Failed length of {name} vector is at least 2')
            self.assertLessEqual(roc_widget[0].shape[0], self.binary_data.n_rows + 1,
                                 msg=f'Failed {name} vector no longer than data set')

    def test_score_multiclass_proba_predictor(self):
        scorer = target.MulticlassScoreHandler()
        dtree = tree.DecisionTreeClassifier()
        X, y_true = self.multiclass_data.X, self.multiclass_data.y
        dtree.fit(X, y_true)
        y_predicted_soft = dtree.predict_proba(X)
        y_predicted_hard = dtree.predict(X)
        scores = scorer.score(y_true=y_true,
                              y_predicted_soft=y_predicted_soft,  # type: ignore
                              y_predicted_hard=y_predicted_hard)
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), self.multiclass_data.n_classes,
                             msg=f'Failed length of {name} list = {self.multiclass_data.n_classes}')
            for idx, r in enumerate(roc_widget):
                self.assertGreater(r.shape[0], 2,
                                   msg=f'Failed length of {name}[{idx}] vector is at least 2')
                self.assertLessEqual(r.shape[0], self.multiclass_data.n_rows + 1,
                                     msg=f'Failed {name}[{idx}] vector no longer than data set')

    @skip('Decision function not supported for multiclass roc_auc.')
    def test_score_multiclass_decision_predictor(self):
        scorer = target.MulticlassScoreHandler()
        svc = svm.SVC()
        X, y_true = self.multiclass_data.X, self.multiclass_data.y
        svc.fit(X, y_true)
        y_predicted_soft = svc.decision_function(X)
        y_predicted_hard = svc.predict(X)
        scores = scorer.score(y_true=y_true,
                              y_predicted_soft=y_predicted_soft,
                              y_predicted_hard=y_predicted_hard)
        self.assertGreaterEqual(scores.auc, 0)
        self.assertLessEqual(scores.auc, 1)
        self.assertGreaterEqual(scores.f1, 0)
        self.assertLessEqual(scores.f1, 1)
        self.assertGreaterEqual(scores.accuracy, 0)
        self.assertLessEqual(scores.accuracy, 1)
        for name, roc_widget in [
            ('thresholds', scores.roc_thresholds),
            ('fpr', scores.roc_fpr),
            ('tpr', scores.roc_tpr),
        ]:
            self.assertIsInstance(roc_widget, list,
                                  msg=f'Failed type of {name} is list')
            self.assertEqual(len(roc_widget), self.multiclass_data.n_classes,
                             msg=f'Failed length of {name} list = {self.multiclass_data.n_classes}')
            for idx, r in enumerate(roc_widget):
                self.assertGreater(r.shape[0], 2,
                                   msg=f'Failed length of {name}[{idx}] vector is at least 2')
                self.assertLessEqual(r.shape[0], self.multiclass_data.n_rows + 1,
                                     msg=f'Failed {name}[{idx}] vector no longer than data set')

    def test_add_preprocessor_sparse_data(self):
        dataset = self.all_datasets['sparse'].dataset
        engine = target.MLExperimentEngine(self._default_config,
                                           dataset=dataset,
                                           dataset_base_name='random_sparse_data')
        self.assertEquals(engine.preprocessors, [])
        engine.set_model_by_name('SVC')
        engine.setup_model_pre_post_processors(dataset=dataset)
        self.assertEquals(len(engine.preprocessors), 1)
        self.assertEquals(engine.preprocessors[0][0], 'ScaleData:MaxAbs')
        self.assertIsInstance(engine.preprocessors[0][1], MaxAbsScaler)

    def test_add_preprocessor_requires_nonnegative_data(self):
        dataset = self.nonnegative_data.dataset
        engine = target.MLExperimentEngine(self._default_config,
                                           dataset=dataset,
                                           dataset_base_name='random_nonneg_data')
        self.assertEquals(engine.preprocessors, [])
        engine.set_model_by_name('CategoricalNB')
        engine.setup_model_pre_post_processors(dataset=dataset)
        self.assertEquals(len(engine.preprocessors), 1)
        self.assertEquals(engine.preprocessors[0][0], 'ScaleData:MinMax')
        self.assertIsInstance(engine.preprocessors[0][1], MinMaxScaler)

    def test_add_preprocessor_general(self):
        dataset = self.multiclass_data.dataset
        engine = target.MLExperimentEngine(self._default_config,
                                           dataset=dataset,
                                           dataset_base_name='random_geneal_data')
        self.assertEquals(engine.preprocessors, [])
        engine.set_model_by_name('SVC')
        engine.setup_model_pre_post_processors(dataset=dataset)
        self.assertEquals(len(engine.preprocessors), 1)
        self.assertEquals(engine.preprocessors[0][0], 'ScaleData:Std')
        self.assertIsInstance(engine.preprocessors[0][1], StandardScaler)

    def test_engine_end_to_end_all_classifiers(self):
        pass


if __name__ == '__main__':
    main()
