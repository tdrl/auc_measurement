"""Unit tests for version.py."""

from unittest import TestCase, main
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree

import auc_measurement.version as target


class TestVersion(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._temp_dir = Path(mkdtemp())

    def tearDown(self) -> None:
        rmtree(self._temp_dir)
        return super().tearDown()

    def test_version_is_semantic(self):
        print(f'version = {target.get_version()}')
        self.assertRegex(target.get_version(), r'\d+\.\d+\.\d+')


if __name__ == '__main__':
    main()
