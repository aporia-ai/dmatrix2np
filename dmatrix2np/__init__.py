from .dmatrix_to_numpy import dmatrix_to_numpy
from .exceptions import DMatrix2NpError, InvalidStructure, UnsupportedVersion, InvalidInput

# Single source the package version
try:
    # Python 3.8 - in std lib
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    # Other python versions
    from importlib_metadata import version, PackageNotFoundError  # type: ignore
try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
