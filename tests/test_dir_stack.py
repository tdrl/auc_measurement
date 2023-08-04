"""Test suite for the basic dir_stack."""


from unittest import TestCase, main
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree

from auc_measurement.dir_stack import dir_stack_push


class TestDirStack(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._test_dir = Path(mkdtemp())
        self._test_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self) -> None:
        rmtree(self._test_dir)
        return super().tearDown()
    
    def assertSamePath(self, path_a: Path, path_b: Path):
        self.assertTrue(path_a.samefile(path_b),
                        f"Expected '{path_a}' to resolve to the same filesystem "
                        f"object as '{path_b}', but they didn't.")
        
    def test_pushd_cwd(self):
        cwd = Path.cwd()
        with dir_stack_push(Path.cwd()) as here:
            self.assertSamePath(cwd, here)

    def test_pushd_new_dir(self):
        start = Path.cwd()
        new_dir = self._test_dir / 'some_place'
        with dir_stack_push(new_dir, force_create=True) as here:
            self.assertSamePath(new_dir, here)
            self.assertSamePath(new_dir, Path.cwd())
        self.assertSamePath(Path.cwd(), start)

    def test_pushd_error_on_nonexistent_dir(self):
        start = Path.cwd()
        new_dir = self._test_dir / 'nonexistent'
        with self.assertRaises(FileNotFoundError):
            with dir_stack_push(new_dir, force_create=False) as here:
                pass
        self.assertSamePath(Path.cwd(), start)


if __name__ == '__main__':
    main()