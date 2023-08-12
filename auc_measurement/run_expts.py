"""Main experiment runner rig.
"""

from pathlib import Path
from sys import argv
import json
from typing import Dict, Union, List, Tuple
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import logging
import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.utils import Bunch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
from sklearn.utils.multiclass import type_of_target
from joblib import dump  # type: ignore
from datetime import datetime

from auc_measurement.dir_stack import dir_stack_push
from auc_measurement.registries import DATA_LOADER_REGISTRY, MODEL_REGISTRY
from auc_measurement.scores import Scores
from auc_measurement.version import get_version


# Type definition for a sequence of preprocessors.
PreprocessorChain = List[Tuple[str, TransformerMixin]]


@dataclass_json
@dataclass
class ExptParams:
    folds: int = 10


@dataclass_json
@dataclass
class Config:
    experiments_output_dir: str
    random_seed: int = 3263827
    large_data_threshold: int = 10000
    datasets: List[str] = field(default_factory=list)
    models_to_test: Dict[str, Dict[str, Union[str, int, float, bool]]] = field(default_factory=dict)
    small_data: ExptParams = field(default_factory=ExptParams)
    large_data: ExptParams = field(default_factory=ExptParams)


def load_config(config_fname: Union[str, Path]) -> Config:
    # Note: There _seems_ to be a bug in dataclass_json that infers the wrong type
    # for a highly nested field, so it fails to deserialze bool model parameters
    # correctly. We'll skip this for the moment until/unless we need such a param.
    with open(config_fname, 'r') as raw:
        return Config.schema().loads(raw.read())  # type: ignore


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
    else:
        # Binary class => (n,) shape. Insert a pseudo-dimension for uniformity => (n, 1) dim.
        y_true = np.expand_dims(y_true, axis=1)
        y_predicted = [y_predicted]
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


def do_predict(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """Figure out whether to use decision_function or predict_proba.

    Args:
        model (_type_): Classifier model.
        X (_type_): Data feature set.

    Returns:
        _type_: Predicted confidence/margin/soft values.
    """
    if hasattr(model, 'decision_function'):
        return model.decision_function(X)  # type: ignore
    else:
        return model.predict_proba(X)  # type: ignore


def run_one_model(X, y, cv_splits, model):
    for idx, (train, test) in enumerate(cv_splits):
        logging.info(f'    Doing split {idx}')
        with dir_stack_push(Path.cwd() / f'fold_{idx}', force_create=True):
            if is_complete():
                logging.info(f'    split {idx} already done; skipping.')
                continue
            logging.info(f'    Fitting model {model}...')
            dump(train, 'train_indices.joblib')
            dump(test, 'test_indices.joblib')
            model.fit(X[train], y[train])
            logging.info('    Done.')
            dump(model, 'model.joblib')
            dump(y[test], 'ground_truth_labels.joblib')
            logging.info('    Predicting on test fold...')
            y_predicted = do_predict(model=model, X=X[test])
            dump(y_predicted, 'predictions.joblib')
            scores = score_predictions(y_true=y[test], y_predicted=y_predicted)
            with open('final_scores.json', 'w') as scores_out:
                scores_out.write(scores.to_json(indent=2))  # type: ignore
            logging.info('    Done.')
            mark_complete()


def run_single_expt(config: Config, dataset: Bunch, preprocessors: PreprocessorChain = []):
    if is_complete():
        return
    X = dataset['data']
    y = dataset['target']
    rows = X.shape[0]
    if rows < config.large_data_threshold:
        expt_config = config.small_data
        logging.info(f'  Dataset has {rows} points; considered SMALL.')
    else:
        expt_config = config.large_data
        logging.info(f'  Dataset has {rows} points; considered LARGE.')
    cv = StratifiedShuffleSplit(n_splits=expt_config.folds, random_state=config.random_seed)
    cv_splits = cv.split(X, y)
    for model_name, params in config.models_to_test.items():
        with dir_stack_push(Path.cwd() / model_name, force_create=True):
            logging.info(f'  Doing model {model_name}...')
            if is_complete():
                logging.info(f'  Model {model_name} already done; skipping.')
                continue
            model = MODEL_REGISTRY[model_name](**params, random_state=config.random_seed)
            with open('model_params.json', 'w') as params_out:
                json.dump(model.get_params(), params_out, indent=2)
            pipeline = Pipeline(preprocessors + [(f'Model_{model_name}', model)])
            run_one_model(X=X, y=y, cv_splits=cv_splits, model=pipeline)
            mark_complete()
    mark_complete()


def run_one_dataset(config: Config, dataset_name: str, dataset: Bunch, preprocessors: PreprocessorChain = []):
    with dir_stack_push(Path.cwd() / dataset_name, force_create=True) as expt_dir:
        if is_complete():
            logging.info(f'Dataset {dataset_name} already done; skipping.')
            return
        logging.info(f'Created experiment directory for dataset {dataset_name} at {expt_dir}')
        with open('dataset_descr.txt', 'w') as descr_out:
            descr_out.write(dataset['DESCR'])
        with open('dataset_name.txt', 'w') as name_out:
            name_out.write(dataset_name + '\n')
        with open('preprocessing.txt', 'w') as preproc_out:
            preproc_out.writelines([p[0] for p in preprocessors])
        run_single_expt(config=config, dataset=dataset, preprocessors=preprocessors)
        mark_complete()


def run_all_expts(config: Config):
    for dataset_name in config.datasets:
        logging.info(f'===== Doing {dataset_name} =====')
        dataset = DATA_LOADER_REGISTRY[dataset_name]()
        y_type = type_of_target(dataset.target)
        if y_type not in ('binary', 'multiclass'):
            logging.warning(f'Dataset {dataset_name} is neither binary nor multiclass; skipping.')
            continue
        preprocessors: List[Tuple[str, TransformerMixin]] = [('PreprocessScaleData', StandardScaler())]
        if y_type == 'multiclass':
            logging.info(f'Dataset {dataset_name} is multiclass; training both raw and one-vs-all')
            run_one_dataset(config, dataset_name + '_raw', dataset, preprocessors)
            preprocessors.append(('BinarizeLabels', LabelBinarizer()))
            run_one_dataset(config, dataset_name + '_one_v_all', dataset, preprocessors)


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
