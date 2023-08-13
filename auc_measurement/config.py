"""Configuration information data structure and loader."""

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from pathlib import Path
from typing import List, Dict, Union


@dataclass_json
@dataclass
class ExptParams:
    """Parameters governing a single, data size-dependent experiment."""
    folds: int = 10  # Number of folds to do in the highest level split.
    calibration_folds: int = 4  # Number of folds to use in fitting the probability calibrator.
    calibration_type: str = 'sigmoid'  # 'sigmoid' or 'isotonic'. See docs on CalibratedClassifierCV.


@dataclass_json
@dataclass
class Config:
    """Parameters governing an overall experiment run."""
    experiments_output_dir: str
    random_seed: int = 3263827
    large_data_threshold: int = 10000
    datasets: List[str] = field(default_factory=list)
    models_to_test: Dict[str, Dict[str, Union[str, int, float, bool]]] = field(default_factory=dict)
    small_data: ExptParams = field(default_factory=ExptParams)
    large_data: ExptParams = field(default_factory=ExptParams)


def load_config(config_fname: Union[str, Path]) -> Config:
    """Load a configuration from file.

    Args:
        config_fname (Union[str, Path]): Location to load from.

    Returns:
        Config: Configuration.
    """
    # Note: There _seems_ to be a bug in dataclass_json that infers the wrong type
    # for a highly nested field, so it fails to deserialze bool model parameters
    # correctly. We'll skip this for the moment until/unless we need such a param.
    with open(config_fname, 'r') as raw:
        return Config.schema().loads(raw.read())  # type: ignore
