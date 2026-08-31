from importlib import metadata

from .api import DiffMemory, create_memory_interface

# Single source of truth for the version is pyproject.toml; installed builds
# resolve it via package metadata. Source-tree runs without installation get
# an explicit dev marker (never a hardcoded release literal that could drift).
try:
    __version__ = metadata.version("diffmem")
except Exception:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+source"

__all__ = ["DiffMemory", "create_memory_interface", "__version__"]
