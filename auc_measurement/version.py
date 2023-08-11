"""Helper to retrieve a uniform version number.

This roots into the 'version' field set in the pyproject.toml file."""

from pathlib import Path
from toml import load


_VERSION = None


def get_version() -> str:
    """Return the semantic version of this code, as set in pypoetry.toml."""
    global _VERSION
    if _VERSION is None:
        pyproject_toml_file = Path(__file__).parent.parent / 'pyproject.toml'
        pyproject_dict = load(pyproject_toml_file)
        _VERSION = pyproject_dict['tool']['poetry']['version']
    return _VERSION
