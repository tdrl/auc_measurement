"""Unit tests for ml_handlers.py."""

from unittest import TestCase, main
from pathlib import Path
from tempfile import mkdtemp
from os import chdir
from shutil import rmtree
from sklearn.datasets import make_classification
from sklearn.utils import Bunch

from auc_measurement.config import Config
import auc_measurement.ml_handlers as target


class TestMlHandlers(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._origin = Path.cwd()
        self._temp_dir = Path(mkdtemp())
        chdir(self._temp_dir)

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
        handler = target.data_split_handler_factory(config=config, dataset=dataset)
        self.assertEqual(handler._n_splits, 5)
        splits = handler.split_data(dataset.data, dataset.target)
        self.assertEqual(len(list(splits)), 5)

    def test_data_split_handler_recognizes_large_data(self):
        config = Config(experiments_output_dir='.', large_data_threshold=37)
        config.small_data.folds = 5
        config.large_data.folds = 2
        X, y = make_classification(n_samples=50)
        dataset = Bunch(data=X, target=y)
        handler = target.data_split_handler_factory(config=config, dataset=dataset)
        self.assertEqual(handler._n_splits, 2)
        splits = handler.split_data(dataset.data, dataset.target)
        self.assertEqual(len(list(splits)), 2)

    def test_add_remove_preprocessors(self):
        config = Config(experiments_output_dir='.')
        X, y = make_classification(n_samples=20)
        dataset = Bunch(data=X, target=y)
        handlers = target.MLExperimentHandlerSet(config=config, dataset=dataset, dataset_base_name='testData')
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


if __name__ == '__main__':
    main()
