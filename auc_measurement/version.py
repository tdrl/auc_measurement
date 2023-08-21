"""Helper to retrieve a uniform version number.

This roots into the 'version' field set in the pyproject.toml file."""

from importlib.resources import files as resource_files
import json
from pathlib import Path
from toml import load
from typing import Dict


_VERSION = None


def get_version() -> str:
    """Return the semantic version of this code, as set in pypoetry.toml."""
    global _VERSION
    if _VERSION is None:
        pyproject_toml_file = Path(__file__).parent.parent / 'pyproject.toml'
        pyproject_dict = load(pyproject_toml_file)
        _VERSION = pyproject_dict['tool']['poetry']['version']
    return _VERSION


def get_git_info() -> Dict[str, str]:
    """Return a best guess at git info for the executing program.

    Assuming that the code is being run directly from a git-controlled directory, and
    that the `git` executable is availble on the ${PATH}, this runs git to retrieve and
    return info about the current branch, checkpoint, and origin repo.

    Returns:
        Dict[str, str]: Dictionary of git info, with the keys:
            - 'repo': Remote origin URL
            - 'branch': Current branch
            - 'commit-hash': Git commit hash for HEAD
    """
    try:
        result = json.loads(resource_files('auc_measurement').joinpath('resources/git_info.json').read_text())
    except IOError:
        result = {
            'repo': '<unknown>',
            'branch': '<unknown>',
            'commit-hash': '<unknown>',
        }
    return result
