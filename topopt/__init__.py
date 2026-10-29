"""topopt: general 3D topology optimization (see CONTRACTS.md)."""
from importlib import metadata as _metadata

try:
    __version__ = _metadata.version("topopt")
except _metadata.PackageNotFoundError:   # source tree that is not installed
    __version__ = "unknown"
