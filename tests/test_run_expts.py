"""Test suite for the main experiment runner rig."""

from unittest import TestCase, main
from tempfile import mkdtemp
from shutil import rmtree
from pathlib import Path
from marshmallow.exceptions import ValidationError
import json

import auc_measurement.run_expts as target
from auc_measurement.dir_stack import dir_stack_push


class TestRunExpts(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        rmtree(self._temp_dir)
        return super().tearDown()

    def test_load_config_minimal_required(self):
        config_file = self._temp_dir / 'test_config.json'
        with open(config_file, 'w') as config_out:
            config_out.write(f"""{{
                "experiments_output_dir": "{self._temp_dir / 'expt_dir'}"
            }}""")
        config = target.load_config(config_file)
        self.assertEqual(config.experiments_output_dir, str(self._temp_dir / 'expt_dir'))
        self.assertEqual(config.random_seed, 3263827)
        self.assertEqual(len(config.datasets), 0)

    def test_load_config_incomplete(self):
        config_file = self._temp_dir / 'test_config.json'
        with open(config_file, 'w') as config_out:
            config_out.write("""{}""")
        with self.assertRaises(ValidationError):
            _ = target.load_config(config_file)

    def test_load_config_complete(self):
        config_file = self._temp_dir / 'test_config.json'
        with open(config_file, 'w') as config_out:
            config_out.write(f"""{{
                "experiments_output_dir": "{self._temp_dir / 'expt_dir'}",
                "random_seed": 421,
                "large_data_threshold": 500,
                "datasets": ["Luke", "Leia", "Han"],
                "models_to_test": {{
                    "R2-D2": {{ "height": 1.08, "class": "astromech" }},
                    "C-3PO": {{ "height": 1.77, "class": "protocol" }}
                }},
                "small_data": {{
                    "folds": 13
                }},
                "large_data": {{
                    "folds": 4
                }}
            }}""")
        config = target.load_config(config_file)
        self.assertEqual(config.experiments_output_dir, str(self._temp_dir / 'expt_dir'))
        self.assertEqual(config.random_seed, 421)
        self.assertEqual(config.datasets, ['Luke', 'Leia', 'Han'])
        self.assertEqual(config.models_to_test['R2-D2']['height'], 1.08)
        self.assertEqual(config.models_to_test['C-3PO']['class'], 'protocol')
        self.assertEqual(config.small_data.folds, 13)
        self.assertEqual(config.large_data.folds, 4)

    def test_mark_complete(self):
        with dir_stack_push(self._temp_dir):
            target.mark_complete()
            with open('.complete', 'r') as c_in:
                complete_data = json.load(c_in)
            self.assertRegex(complete_data['timestamp'], r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z.*')
            self.assertRegex(complete_data['version'], r'\d+\.\d+.\d+')
        


if __name__ == '__main__':
    main()
