"""Main experiment runner rig.
"""

from pathlib import Path
from sys import argv
from config_schema import get_config_schema
import json
from jsonschema import validate
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import logging
from sklearn import (
    svm,
    tree
)
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.utils import Bunch
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import dump

from dir_stack import dir_stack_push


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
    small_data: ExptParams = field(default_factory=ExptParams)
    large_data: ExptParams = field(default_factory=ExptParams)


def load_config(config_fname: str) -> Config:
    schema = get_config_schema()
    with open(config_fname, 'r') as raw:
        config = json.load(raw)
        validate(config, schema)
        return Config(**config)


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
    for idx, (train, test) in enumerate(cv.split(X, y)):
        logging.info(f'  Doing split {idx}')
        with dir_stack_push(Path.cwd() / f'fold_{idx}', force_create=True):
            model = svm.SVC(random_state=config.random_seed, probability=True)
            with open('model_params.json', 'w') as params_out:
                json.dump(model.get_params(), params_out, indent=2)
            logging.info(f'  Fitting model {model}...')
            model.fit(X[train], y[train])
            logging.info('  Done.')
            dump(model, 'model.joblib')
            dump(y[test], 'ground_truth_labels.joblib')
            logging.info('  Predicting on test fold...')
            dump(model.predict_proba(X[test]), 'predictions.joblib')
            logging.info('  Done.')
            Path('.complete').touch()
    Path('.complete').touch()


def run_all_expts(config: Config):
    loaders = [
        load_iris,
        load_diabetes,
        # load_breast_cancer
    ]
    for loader in loaders:
        logging.info(f'===== Doing {loader.__name__} =====')
        dataset_name = loader.__name__.removeprefix('load_')
        with dir_stack_push(Path.cwd() / dataset_name, force_create=True) as expt_dir:
            logging.info(f'Created experiment directory for dataset {dataset_name} at {expt_dir}')
            dataset = loader()
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