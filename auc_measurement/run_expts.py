"""Main experiment runner rig.
"""

from pathlib import Path
from sys import argv
from config_schema import get_config_schema
import json
from jsonschema import validate
from typing import Dict, Union, List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import logging
from sklearn.utils import Bunch
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump

from dir_stack import dir_stack_push
from registries import DATA_LOADER_REGISTRY, MODEL_REGISTRY


@dataclass_json
@dataclass
class ExptParams:
    folds: int = 10


@dataclass_json
@dataclass
class Config:
    experiments_output_dir: str
    random_seed: int = 69
    large_data_threshold: int = 10000
    datasets: List[str] = field(default_factory=list)
    models_to_test: Dict[str, Dict[str, Union[str, int, float, bool]]] = field(default_factory=dict)
    small_data: ExptParams = field(default_factory=ExptParams)
    large_data: ExptParams = field(default_factory=ExptParams)


def load_config(config_fname: str) -> Config:
    # Note: There _seems_ to be a bug in dataclass_json that infers the wrong type
    # for a highly nested field, so it fails to deserialze bool model parameters
    # correctly. We'll skip this for the moment until/unless we need such a param.
    with open(config_fname, 'r') as raw:
        return Config.schema().loads(raw.read())


def run_one_model(X, y, cv_splits, model):
    for idx, (train, test) in enumerate(cv_splits):
        logging.info(f'    Doing split {idx}')
        with dir_stack_push(Path.cwd() / f'fold_{idx}', force_create=True):
            logging.info(f'    Fitting model {model}...')
            model.fit(X[train], y[train])
            logging.info('    Done.')
            dump(model, 'model.joblib')
            dump(y[test], 'ground_truth_labels.joblib')
            logging.info('    Predicting on test fold...')
            dump(model.predict_proba(X[test]), 'predictions.joblib')
            logging.info('    Done.')
            Path('.complete').touch()


def run_single_expt(config: Config, dataset: Bunch):
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
            model = MODEL_REGISTRY[model_name](**params, random_state=config.random_seed)
            with open('model_params.json', 'w') as params_out:
                json.dump(model.get_params(), params_out, indent=2)
            run_one_model(X=X, y=y, cv_splits=cv_splits, model=model)
    Path('.complete').touch()


def run_all_expts(config: Config):
    for dataset_name in config.datasets:
        logging.info(f'===== Doing {dataset_name} =====')
        with dir_stack_push(Path.cwd() / dataset_name, force_create=True) as expt_dir:
            logging.info(f'Created experiment directory for dataset {dataset_name} at {expt_dir}')
            dataset = DATA_LOADER_REGISTRY[dataset_name]()
            with open('dataset_descr.txt', 'w') as descr_out:
                descr_out.write(dataset['DESCR'])
            with open('dataset_name.txt', 'w') as name_out:
                name_out.write(dataset_name + '\n')
            run_single_expt(config=config, dataset=dataset)


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
        with open('config.json', 'w') as config_out:
            config_out.write(config.to_json(indent=2))
        run_all_expts(config=config)


if __name__ == '__main__':
    main()