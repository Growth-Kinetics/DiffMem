# DiffMem: Differential Memory Backend
# Main API exports for direct import into chat agents

from .api import DiffMemory, create_memory_interface, quick_search

__version__ = "0.1.0"
__all__ = ["DiffMemory", "create_memory_interface", "quick_search"]
