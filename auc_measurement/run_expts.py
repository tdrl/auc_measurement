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
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_diabetes,
)
from sklearn.utils import Bunch
from dir_stack import dir_stack_push


@dataclass_json
@dataclass
class ExptParams:
    folds: int = 10


@dataclass_json
@dataclass
class Config:
    experiments_output_dir: str
    large_data_threshold: int = 10000
    small_data: ExptParams = field(default_factory=ExptParams)


def load_config(config_fname: str) -> Config:
    schema = get_config_schema()
    with open(config_fname, 'r') as raw:
        config = json.load(raw)
        validate(config, schema)
        return Config(**config)


def run_single_expt(config: Config, dataset: Bunch):
    pass



def run_all_expts(config: Config):
    loaders = [
        load_iris,
        load_diabetes,
        load_breast_cancer
    ]
    for loader in loaders:
        logging.info('Doing %s', loader.__name__)
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