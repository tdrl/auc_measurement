"""A directory stack context manager, in the style offered by many CLI shells."""

# Note: Could use contextlib.chdir() if we were on Python 3.11. Ah well.

from os import chdir
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Any


@contextmanager
def dir_stack_push(new_dir: Path, force_create: bool = False) -> Generator[Path, Any, Any]:
    cwd = Path.cwd()
    if force_create:
        new_dir.mkdir(parents=True, exist_ok=True)
    try:
        chdir(new_dir)
        yield new_dir
    finally:
        chdir(cwd)
