"""A directory stack context manager, in the style offered by many CLI shells.

Usage:

with dir_stack_push(dest_dir) as new_working_dir:
    # do something in new_working_dir
# restored to previous working dir.
"""

# Note: Could use contextlib.chdir() if we were on Python 3.11. Ah well.

from os import chdir
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Any


@contextmanager
def dir_stack_push(new_dir: Path, force_create: bool = False) -> Generator[Path, Any, Any]:
    """Context manager for a directory stack.

    Args:
        new_dir (Path): Location to change to, while remembering starting dir.
        force_create (bool, optional): Create new_dir if it doesn't already exist. Defaults to False.

    Usage:

    # Will raise FileNotFoundError if dest_dir doesn't exist.
    with dir_stack_push(dest_dir) as new_working_dir:
        # do something in new_working_dir
    # restored to previous working dir.

    or:

    # Will create dest_dir if it doesn't already exist.
    with dir_stack_push(dest_dir, force_create=True) as new_working_dir:
        # do something in new_working_dir
    # restored to previous working dir.        
    """
    cwd = Path.cwd()
    if force_create:
        new_dir.mkdir(parents=True, exist_ok=True)
    try:
        chdir(new_dir)
        yield new_dir
    finally:
        chdir(cwd)
