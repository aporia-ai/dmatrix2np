from .dmatrix_to_numpy import dmatrix_to_numpy
from .exceptions import DMatrix2NpError, InvalidStructure, UnsupportedVersion, InvalidInput

# Single source the package version
try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore
try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
