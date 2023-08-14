"""Main experiment runner rig.
"""

from pathlib import Path
from sys import argv
import json
from typing import Union, List, Mapping, Iterable
import logging
import numpy as np
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import sklearn.metrics as metrics
from sklearn.utils.multiclass import type_of_target
from joblib import dump  # type: ignore
from datetime import datetime

from auc_measurement.config import Config, load_config
from auc_measurement.dir_stack import dir_stack_push
from auc_measurement.ml_handlers import MLExperimentEngine, ExperimentConfigurationException
from auc_measurement.registries import DATA_LOADER_REGISTRY
from auc_measurement.scores import Scores
from auc_measurement.version import get_version


def mark_complete():
    completion_data = {
        'version': get_version(),
        'timestamp': datetime.utcnow().strftime('%FT%TZ'),
    }
    with open('.complete', 'w') as c_out:
        json.dump(completion_data, c_out)


def is_complete() -> bool:
    return Path('.complete').exists()


def score_predictions(y_true: np.ndarray, y_predicted: Union[np.ndarray, List[np.ndarray]]) -> Scores:
    if type_of_target(y_true) == 'multiclass':
        # We've trained in multiclass mode, but for ROC we need binary data. So we'll do
        # one-vs-all for each label.
        y_true = LabelBinarizer().fit(y_true).transform(y_true)  # type: ignore
    elif type_of_target(y_true) == 'binary':
        # Binary class => ground truth is (n,) shape. Insert a pseudo-dimension for uniformity => (n, 1) dim.
        y_true = np.expand_dims(y_true, axis=1)
        # In multiclass, predicted is a list of vectors.
        y_predicted = [y_predicted]
    else:
        raise ExperimentConfigurationException('Not sure how we got to scoring without realizing '
                                               'that this data set is neither binary nor '
                                               'multiclass, but here we are. *shrug*')
    roc_fpr = []
    roc_tpr = []
    roc_thresholds = []
    for one_v_all_label, pred in zip(y_true.T, y_predicted):
        fpr, tpr, thresholds = metrics.roc_curve(y_true=one_v_all_label, y_score=pred)
        roc_fpr.append(fpr)
        roc_tpr.append(tpr)
        roc_thresholds.append(thresholds)
    return Scores(
        auc=float(metrics.roc_auc_score(y_true=y_true, y_score=y_predicted)),
        f1=float(metrics.f1_score(y_true=y_true, y_pred=y_predicted)),
        accuracy=float(metrics.accuracy_score(y_true=y_true, y_pred=y_predicted)),
        roc_fpr=roc_fpr,
        roc_tpr=roc_tpr,
        roc_thresholds=roc_thresholds
    )

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
    if isinstance(params, str) or isinstance(params, int) or isinstance(params, float) or isinstance(params, bool) or params is None:
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
            dump(train, 'train_indices.joblib')
            dump(test, 'test_indices.joblib')
            expt_handlers.fit_model(X[train], y[train])
            logging.info('    Done.')
            dump(expt_handlers.model, 'model.joblib')
            dump(y[test], 'ground_truth_labels.joblib')
            logging.info('    Predicting on test fold...')
            y_predicted = expt_handlers.predict_soft(X=X[test])
            dump(y_predicted, 'predictions.joblib')
            scores = score_predictions(y_true=y[test], y_predicted=y_predicted)
            with open('final_scores.json', 'w') as scores_out:
                scores_out.write(scores.to_json(indent=2))  # type: ignore
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
            with open('model_params.json', 'w') as params_out:
                json.dump(fully_expand_params(expt_handlers.model), params_out, indent=2)  # type:ignore
            run_one_model(X=X, y=y, expt_handlers=expt_handlers)
            mark_complete()
    mark_complete()


def run_one_dataset(config: Config, expt_handlers: MLExperimentEngine, dataset_name: str, dataset: Bunch):
    with dir_stack_push(Path.cwd() / dataset_name, force_create=True) as expt_dir:
        if is_complete():
            logging.info(f'  Dataset {dataset_name} already done; skipping.')
            return
        logging.info(f'  Created experiment directory for dataset {dataset_name} at {expt_dir}')
        with open('dataset_descr.txt', 'w') as descr_out:
            descr_out.write(dataset['DESCR'])
        with open('dataset_name.txt', 'w') as name_out:
            name_out.write(dataset_name + '\n')
        with open('preprocessing.txt', 'w') as preproc_out:
            preproc_out.writelines([p[0] for p in expt_handlers.preprocessors])
        run_single_expt(config=config, expt_handlers=expt_handlers, dataset=dataset)
        mark_complete()


def run_all_expts(config: Config):
    for dataset_name in config.datasets:
        logging.info(f'===== Doing {dataset_name} =====')
        dataset = DATA_LOADER_REGISTRY[dataset_name]()
        try:
            expt_handlers = MLExperimentEngine(config=config, dataset=dataset, dataset_base_name=dataset_name)
            expt_handlers.add_preprocessor('ScaleData', StandardScaler())
            if expt_handlers.y_type == 'multiclass':
                logging.info(f'  Dataset {dataset_name} is multiclass; training both raw and one-vs-all')
                run_one_dataset(config, expt_handlers, dataset_name + '_raw', dataset)
                expt_handlers.add_preprocessor('BinarizeLabels', LabelBinarizer())
                run_one_dataset(config, expt_handlers, dataset_name + '_one_v_all', dataset)
                expt_handlers.remove_preprocessor('BinarizeLabels')
        except ExperimentConfigurationException:
            pass


def main(config=None):
    # TODO(hlane) Configure logging smarter.
    logging.basicConfig(format='{levelname}|{asctime}] {message}',
                        level=logging.INFO,
                        style='{',
                        datefmt='%Y-%m-%d:%H:%M:%S')
    logging.info('Starting.')
    if config is None:
        try:
            config = load_config(argv[1])
        except IndexError:
            logging.error('Usage: python run_expts config_file.json')
            raise
    expt_dir = Path(config.experiments_output_dir)
    logging.info('Creating experiment output dir at %s',
                 config.experiments_output_dir)
    with dir_stack_push(expt_dir, force_create=True) as _:
        if is_complete():
            logging.info('All experiments already marked done. Stopping immediately.')
            return
        with open('config.json', 'w') as config_out:
            config_out.write(config.to_json(indent=2))  # type: ignore
        run_all_expts(config=config)
        mark_complete()


if __name__ == '__main__':
    main()
