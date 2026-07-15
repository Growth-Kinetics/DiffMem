# CAPABILITY: Agent/strategy registry — external extension seam (ADR-003).
# INPUTS:  strategy name (str) for lookup; none for listing.
# OUTPUTS: StrategyInfo metadata (list_strategies) or a loaded strategy object
#          (get_strategy) discovered via entry-point group "diffmem.strategies".
# CONSTRAINTS:
#   - Core NEVER imports an external strategy package by name; discovery is
#     purely metadata-driven via importlib.metadata.
#   - Built-ins are registered under the SAME group in pyproject.toml, so
#     list_strategies() is non-empty with OSS alone and runtime behaviour is
#     identical whether or not an external package is installed.
#   - entry_points(group=...) is the 3.10+ keyword API; this package floors at 3.10.
"""Discoverable strategy registry for DiffMem.

Strategies (agents, ontology providers, etc.) are registered under the
entry-point group ``diffmem.strategies``. The OSS package registers its own
built-ins there, so behaviour is unchanged when no external package is present.
External packages (e.g. the private ``diffmem-pro``) add entries via the same
group; the core resolves them by name without importing the package directly.
"""
import logging
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "diffmem.strategies"


class StrategyNotFoundError(LookupError):
    """Raised by get_strategy() for a name no package has registered.

    A defined error (not a bare KeyError) so callers can distinguish a missing
    strategy from unrelated lookup failures.
    """


@dataclass(frozen=True)
class StrategyInfo:
    """Metadata for a discoverable strategy. Reading it never imports the target."""

    name: str
    value: str  # "module:attr" target string, as declared in the entry point


def _entries() -> list[Any]:
    """Raw entry points for our group (fresh read each call; cheap metadata lookup)."""
    return list(entry_points(group=ENTRY_POINT_GROUP))


def list_strategies() -> list[StrategyInfo]:
    """All registered strategies as metadata, sorted by name for deterministic order.

    Does NOT load any target — safe to call to enumerate available strategies
    without triggering imports or side effects.
    """
    infos = [StrategyInfo(name=e.name, value=e.value) for e in _entries()]
    return sorted(infos, key=lambda s: s.name)


def get_strategy(name: str) -> Any:
    """Load and return the strategy object registered under ``name``.

    Raises ``StrategyNotFoundError`` if nothing is registered under ``name`` —
    never returns None and never raises a bare KeyError.
    """
    for entry in _entries():
        if entry.name == name:
            loaded = entry.load()
            logger.info("STRATEGY_RESOLVED: name=%s target=%s", name, entry.value)
            return loaded
    known = [s.name for s in list_strategies()]
    raise StrategyNotFoundError(
        f"No strategy named {name!r} in group {ENTRY_POINT_GROUP!r}. Known: {known}"
    )
