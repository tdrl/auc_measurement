"""Helper to retrieve a uniform version number.

This roots into the 'version' field set in the pyproject.toml file."""

from pathlib import Path
from toml import load
from typing import Dict
import subprocess


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
            - 'hash': Git commit hash for HEAD
    """
    try:
        repo_info = subprocess.run(args=['git', 'config', '--get',' remote.origin.url'],
                                   capture_output=True,
                                   text=True)
        repo_info.check_returncode()
        branch_info = subprocess.run(args=['git', 'branch', '--show-current'],
                                     capture_output=True,
                                     text=True)
        branch_info.check_returncode()
        commit_hash_info = subprocess.run(args=['git', 'rev-parse', 'HEAD'],
                                          capture_output=True,
                                          text=True)
        commit_hash_info.check_returncode()
        result = {
            'repo': repo_info.stdout.strip(),
            'branch': branch_info.stdout.strip(),
            'hash': commit_hash_info.stdout.strip(),
        }
    except (IOError, subprocess.CalledProcessError):
        result = {
            'repo': '<unknown>',
            'branch': '<unknown>',
            'hash': '<unknown>',
        }
    return result
