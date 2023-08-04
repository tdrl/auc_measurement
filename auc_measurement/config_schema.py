"""Only contains the schema for the config file."""

from typing import Dict, Any

_config_schema = {
    '$id': 'https://github.com/tdrl/auc_measurement/tree/main/auc_measurement/config_schema.py',
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'description': 'Schema for AUC measurement config file.',
    'type': 'object',
    'properties': {
        'experiments_output_dir': {
            'description': 'Destination directory for experiments. Created if it doesn''t exist already.',
            'type': 'string',
        },
        'random_seed': {
            'description': 'An integer random seed value, as used by sklearn functions.',
            'type': 'integer',
        },
        'large_data_threshold': {
            'description': 'How many rows for a dataset to be considered "large". Large datasets will receive fewer xval folds than small ones.',
            'type': 'integer',
        },
        'small_data': {
            'description': 'Parameters that apply to small data sets.',
            'type': 'object',
            'properties': {
                'folds': {
                    'description': 'Number of cross-validation folds for test.',
                    'type': 'integer',
                }
            }
        }
    },
    'required': [
        'experiments_output_dir'
    ]
}


def get_config_schema() -> Dict[str, Any]:
    """Return the jsonschema object representing the configuration file.

    Returns:
        dict: Nested dict of jsonschema description
    """
    return _config_schema