"""Unit tests for ml_handlers.py."""

from unittest import TestCase, main
from pathlib import Path
from tempfile import mkdtemp
from os import chdir
from shutil import rmtree
from sklearn.datasets import make_classification
from sklearn.utils import Bunch
import numpy as np
from numpy.testing import assert_array_almost_equal

from auc_measurement.config import Config
import auc_measurement.ml_handlers as target
import auc_measurement.registries as reg


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
        self._default_binary_rows = 40
        X_bin, y_bin = make_classification(n_samples=self._default_binary_rows, n_classes=2)
        self._default_binary_dataset = Bunch(data=X_bin, target=y_bin)
        self._default_multiclass_rows = 100
        self._default_multiclass_classes = 7
        X_multi, y_multi = make_classification(n_samples=self._default_multiclass_rows,
                                               n_classes=self._default_multiclass_classes,
                                               n_features=11,
                                               n_informative=9)
        self._default_multiclass_dataset = Bunch(data=X_multi, target=y_multi)

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
        experiment = target.MLExperimentHandlerSet(config=config, dataset=dataset, dataset_base_name='testData')
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
        experiment = target.MLExperimentHandlerSet(config=config, dataset=dataset, dataset_base_name='testData')
        handler = experiment.data_split_handler_factory()
        self.assertEqual(handler._n_splits, 2)
        splits = handler.split_data(dataset.data, dataset.target)
        self.assertEqual(len(list(splits)), 2)

    def test_add_remove_preprocessors(self):
        handlers = target.MLExperimentHandlerSet(config=self._default_config,
                                                 dataset=self._default_binary_dataset,
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
        handlers = target.MLExperimentHandlerSet(config=self._default_config,
                                                 dataset=self._default_binary_dataset,
                                                 dataset_base_name='testData')
        with self.assertRaisesRegex(target.ExperimentConfigurationException, r'blurfle'):
            handlers.set_model_by_name('blurfle')

    def test_predict_soft_error_if_model_unset(self):
        handlers = target.MLExperimentHandlerSet(config=self._default_config,
                                                 dataset=self._default_binary_dataset,
                                                 dataset_base_name='testData')
        with self.assertRaises(ValueError):
            handlers.predict_soft(self._default_binary_dataset.data)

    def test_predict_all_classifiers_binary(self):
        handlers = target.MLExperimentHandlerSet(config=self._default_config,
                                                 dataset=self._default_binary_dataset,
                                                 dataset_base_name='testData')
        for model_name in target.MODEL_REGISTRY:
            handlers.set_model_by_name(model_name)
            handlers.fit_model(self._default_binary_dataset.data, self._default_binary_dataset.target)
            yhat = handlers.predict_soft(self._default_binary_dataset.data)
            self.assertEqual(yhat.shape, (self._default_binary_rows, 2), f'Failed on model = {model_name}')
            assert_array_almost_equal(yhat.sum(axis=1), np.ones((self._default_binary_rows,)),
                                      err_msg=f'Failed on model = {model_name}')

    def test_predict_all_classifiers_multiclass(self):
        handlers = target.MLExperimentHandlerSet(config=self._default_config,
                                                 dataset=self._default_multiclass_dataset,
                                                 dataset_base_name='testData')
        for model_name in target.MODEL_REGISTRY:
            handlers.set_model_by_name(model_name)
            handlers.fit_model(self._default_multiclass_dataset.data, self._default_multiclass_dataset.target)
            yhat = handlers.predict_soft(self._default_multiclass_dataset.data)
            self.assertEqual(yhat.shape, (self._default_multiclass_rows, self._default_multiclass_classes),
                             f'Failed on model = {model_name}')
            assert_array_almost_equal(yhat.sum(axis=1), np.ones((self._default_multiclass_rows,)),
                                      err_msg=f'Failed on model = {model_name}')


if __name__ == '__main__':
    main()
