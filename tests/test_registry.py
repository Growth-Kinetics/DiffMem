# CAPABILITY: Registry seam tests (ADR-003).
# INTENT: Prove the entry-point discovery layer works against the INSTALLED
# package — built-ins surface via list_strategies(), get_strategy resolves and
# loads the built-in consolidator, and unknown names raise the defined error
# rather than returning None. These tests require diffmem to be installed
# (entry points live in dist-info metadata), which `poetry run pytest` ensures.
import pytest

from diffmem.consolidator_agent.agent import ConsolidatorAgent
from diffmem.registry import (
    ENTRY_POINT_GROUP,
    StrategyInfo,
    StrategyNotFoundError,
    get_strategy,
    list_strategies,
)


def test_list_strategies_returns_builtin_consolidator():
    # The OSS package registers its built-in(s) under the same group external
    # packages use, so the registry is non-empty with OSS alone.
    names = [s.name for s in list_strategies()]
    assert "consolidator" in names, f"built-in 'consolidator' missing from {names}"


def test_list_strategies_returns_metadata_without_loading():
    infos = list_strategies()
    assert infos, "expected at least one built-in strategy"
    # Each entry is pure metadata (name + "module:attr" target), not a loaded obj.
    cinfo = next(s for s in infos if s.name == "consolidator")
    assert isinstance(cinfo, StrategyInfo)
    assert cinfo.value == "diffmem.consolidator_agent.agent:ConsolidatorAgent"
    # Deterministic ordering helps callers and downstream tests.
    assert infos == sorted(infos, key=lambda s: s.name)


def test_get_strategy_resolves_and_loads_builtin():
    # get_strategy must actually LOAD the target, proving the built-in
    # registration is live (not just metadata) and resolves to the consolidator.
    loaded = get_strategy("consolidator")
    assert loaded is ConsolidatorAgent


def test_get_strategy_unknown_raises_defined_error():
    # Unknown names raise the registry's own error — never None, never bare KeyError.
    with pytest.raises(StrategyNotFoundError):
        get_strategy("definitely-not-a-registered-strategy-xyz")


def test_entry_point_group_is_stable():
    # The group string is the public contract external packages register against.
    assert ENTRY_POINT_GROUP == "diffmem.strategies"
