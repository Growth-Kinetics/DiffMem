"""
Build a TaskExecutor from the EXECUTOR environment variable.

Mirrors the pattern in storage/factory.py: env-var dispatch, structured
logging, ValueError / NotImplementedError for unknown / unimplemented backends.

Supported values of EXECUTOR (case-insensitive, whitespace stripped):
  inline  — default; runs jobs in the existing ThreadPoolExecutor.
  hatchet — durable queue via Hatchet (implemented in M3).
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from .base import TaskExecutor
from .inline import InlineExecutor

logger = logging.getLogger(__name__)


def build_executor(pool: ThreadPoolExecutor) -> TaskExecutor:
    """Construct and return the configured TaskExecutor.

    Args:
        pool: The existing ThreadPoolExecutor (typically ``_writer_pool``).

    Returns:
        A fully initialised TaskExecutor instance.

    Raises:
        NotImplementedError: If EXECUTOR=hatchet (available in M3).
        ValueError: If EXECUTOR is set to an unknown value.
    """
    raw = os.getenv("EXECUTOR", "inline")
    value = raw.strip().lower()

    if value == "inline":
        logger.info("EXECUTOR_INITIALIZED: type=inline")
        return InlineExecutor(pool)

    if value == "hatchet":
        try:
            from .hatchet import HatchetExecutor  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "HatchetExecutor requires the hatchet extras. "
                "Run: pip install diffmem[hatchet] "
                "(or `poetry install --extras hatchet`)"
            ) from e
        executor = HatchetExecutor(pool)
        logger.info("EXECUTOR_INITIALIZED: type=hatchet")
        return executor

    raise ValueError(
        f"Unknown EXECUTOR={value!r}. Expected: inline, hatchet."
    )
