"""Main experiment runner rig.
"""

from pathlib import Path
from sys import argv
from os import chdir
# from typing import Dict, Union, List
from config_schema import get_config_schema
import json
from jsonschema import validate
from dataclasses import dataclass
import logging
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_diabetes,
)


@dataclass
class Config:
    experiments_output_dir: str


def load_config(config_fname: str) -> Config:
    schema = get_config_schema()
    with open(config_fname, 'r') as raw:
        config = json.load(raw)
        validate(config, schema)
        return Config(**config)


def run_all_expts(config: Config):
    work_dir = Path(config.experiments_output_dir)
    chdir(work_dir)
    loaders = [
        load_iris,
        load_diabetes,
        load_breast_cancer
    ]
    for loader in loaders:
        logging.info('Doing %s', loader.__name__)


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
    expt_dir.mkdir(parents=True, exist_ok=False)
    run_all_expts(config=config)


if __name__ == '__main__':
    main()