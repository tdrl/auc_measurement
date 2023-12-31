"""Main experiment runner rig.
"""

from datetime import datetime
from joblib import dump as dump_joblib  # type: ignore
from logging.config import dictConfig
from pathlib import Path
from sklearn.utils import Bunch
from sys import argv
from typing import Mapping, Iterable, Union, List
import json
import logging
import traceback

from auc_measurement.config import Config, load_config
from auc_measurement.dir_stack import dir_stack_push
from auc_measurement.ml_handlers import MLExperimentEngine, ExperimentConfigurationException, ScoreHandler
from auc_measurement.registries import DATA_LOADER_REGISTRY
from auc_measurement.version import get_version, get_git_info


def dump_json(filename: Union[str, Path], obj: object):
    """Helper function: Save obj as JSON in specified file."""
    with open(filename, 'w') as data_out:
        json.dump(obj=obj, fp=data_out, indent=2)


def dump_text(filename: Union[str, Path], text: Union[str, List[str]]):
    if isinstance(text, str):
        text = [text]
    with open(filename, 'w') as lines_out:
        lines_out.writelines((line + '\n' for line in text))


def mark_complete():
    completion_data = {
        'version': get_version(),
        'timestamp': datetime.utcnow().strftime('%FT%TZ'),
        'git-info': get_git_info(),
    }
    dump_json('.complete', completion_data)


def is_complete() -> bool:
    return Path('.complete').exists()


def fully_expand_params(params):
    """Helper function: Expand all nested objects in get_params() values.

    The issue is that when you call get_params() on an actual estimator, it will return
    a dict of (key, concrete value) pairs. But when you call it on a Pipeline, it doesn't - it
    nests actual instances of models inside the return value. Which is terrible if you want
    to serialize it with JSON. Urgh. So this recurses the data struct, finding things with
    'get_params()' methods and replacing them with their invocation.

    Args:
        params (Any): Object returned by estimator.get_params()
    """
    if (isinstance(params, str) or
        isinstance(params, int) or
        isinstance(params, float) or
        isinstance(params, bool) or
            params is None):
        return params
    if hasattr(params, 'get_params'):
        return fully_expand_params(params.get_params(deep=False))
    if issubclass(type(params), Mapping):
        return {k: fully_expand_params(v) for k, v in params.items()}
    if issubclass(type(params), Iterable):
        return [fully_expand_params(x) for x in params]
    return params


def run_one_model(X, y, expt_handlers: MLExperimentEngine):
    for idx, (train, test) in enumerate(expt_handlers.split_handler.split_data(X, y)):
        logging.info(f'    Doing split {idx}')
        with dir_stack_push(Path.cwd() / f'fold_{idx}', force_create=True):
            if is_complete():
                logging.info(f'    split {idx} already done; skipping.')
                continue
            logging.info(f'    Fitting model {expt_handlers.model_name}...')
            dump_joblib(train, 'train_indices.joblib')
            dump_joblib(test, 'test_indices.joblib')
            expt_handlers.fit_model(X[train], y[train])
            logging.info('    Done.')
            dump_joblib(expt_handlers.model, 'model.joblib')
            dump_joblib(y[test], 'ground_truth_labels.joblib')
            logging.info('    Predicting on test fold...')
            y_predicted_soft = expt_handlers.predict_soft(X=X[test])
            dump_joblib(y_predicted_soft, 'predictions_soft.joblib')
            y_predicted_hard = expt_handlers.predict_hard(X=X[test])
            dump_joblib(y_predicted_hard, 'predictions_hard.joblib')
            scorer = ScoreHandler.score_handler_factory(y[test])
            scores = scorer.score(y_true=y[test], y_predicted_soft=y_predicted_soft, y_predicted_hard=y_predicted_hard)
            dump_text('final_scores.json', scores.to_json(indent=2))  # type: ignore
            logging.info('    Done.')
            mark_complete()


def run_single_expt(config: Config, expt_handlers: MLExperimentEngine, dataset: Bunch):
    if is_complete():
        return
    X = dataset['data']
    y = dataset['target']
    for model_name in config.models_to_test:
        with dir_stack_push(Path.cwd() / model_name, force_create=True):
            logging.info(f'  Doing model {model_name}...')
            if is_complete():
                logging.info(f'  Model {model_name} already done; skipping.')
                continue
            expt_handlers.set_model_by_name(model_name=model_name)
            if not expt_handlers.model_data_are_compatible(dataset=dataset):
                logging.warning(f"Model {model_name} can't handle sparse data in data "
                                f"set {expt_handlers.dataset_base_name} (lame!). Skipping.")
                mark_complete()
                dump_text('.failed',
                          f'Model {model_name} incompatible with data set {expt_handlers.dataset_base_name}')
                continue
            expt_handlers.setup_model_pre_post_processors(dataset=dataset)
            dump_json('model_params.json', fully_expand_params(expt_handlers.model))
            dump_text('preprocessing.txt', [p[0] for p in expt_handlers.preprocessors])
            run_one_model(X=X, y=y, expt_handlers=expt_handlers)
            mark_complete()
    mark_complete()


def run_one_dataset(config: Config, expt_handlers: MLExperimentEngine, dataset_name: str, dataset: Bunch):
    with dir_stack_push(Path.cwd() / dataset_name, force_create=True) as expt_dir:
        if is_complete():
            logging.info(f'  Dataset {dataset_name} already done; skipping.')
            return
        logging.info(f'  Created experiment directory for dataset {dataset_name} at {expt_dir}')
        dump_text('dataset_descr.txt', dataset['DESCR'])
        dump_text('dataset_name.txt', dataset_name)
        run_single_expt(config=config, expt_handlers=expt_handlers, dataset=dataset)
        mark_complete()


def run_all_expts(config: Config):
    for dataset_name in config.datasets:
        logging.info(f'===== Doing {dataset_name} =====')
        dataset = DATA_LOADER_REGISTRY[dataset_name]()
        try:
            expt_handlers = MLExperimentEngine(config=config, dataset=dataset, dataset_base_name=dataset_name)
            run_one_dataset(config, expt_handlers, dataset_name, dataset)
        except ExperimentConfigurationException:
            pass


def main(config=None):
    if config is None:
        try:
            config = load_config(argv[1])
        except IndexError:
            logging.error('Usage: python run_expts config_file.json')
            raise
    expt_dir = Path(config.experiments_output_dir)
    with dir_stack_push(expt_dir, force_create=True):
        if config.logging_config_file is not None:
            with open(config.logging_config_file, 'r') as config_in:
                dictConfig(json.load(config_in))
        else:
            logging.basicConfig(format='{levelname}|{asctime}] {message}',
                                level=logging.INFO,
                                style='{',
                                datefmt='%Y-%m-%d:%H:%M:%S')
        logging.info('Starting.')
        logging.info('Created experiment output dir at %s',
                     config.experiments_output_dir)
        if is_complete():
            logging.info('All experiments already marked done. Stopping immediately.')
            return
        dump_text('config.json', config.to_json(indent=2))
        try:
            run_all_expts(config=config)
        except Exception as e:
            logging.error(f'Fatal exception in execution: {e}')
            logging.error('Stack trace:')
            for frame in traceback.format_tb(e.__traceback__):
                logging.error(f'{frame}')
            raise e
        mark_complete()


if __name__ == '__main__':
    main()
