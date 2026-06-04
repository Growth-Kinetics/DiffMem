# CAPABILITY: Per-worktree lock primitive preventing concurrent consolidator/writer runs.
# INPUTS: Worktree path (Path or str). Reads/writes <worktree>/.diffmem/consolidator.lock.
# OUTPUTS: Context manager; raises LockBusyError on contention with a live PID.
# CONSTRAINTS: POSIX-only (uses os.kill(pid, 0) for liveness probe). Stale-lock reclaim
#              when PID is dead AND lock is older than STALE_AFTER_SECONDS.

from __future__ import annotations

import errno
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

STALE_AFTER_SECONDS = 30 * 60  # 30 minutes


class LockBusyError(RuntimeError):
    """Raised when a live consolidator lock already exists for the worktree."""

    def __init__(self, pid: int, started_at: str):
        self.pid = pid
        self.started_at = started_at
        super().__init__(
            f"LOCK_BUSY: consolidator lock held by pid={pid} started_at={started_at}"
        )


def _pid_alive(pid: int) -> bool:
    """POSIX liveness probe. os.kill(pid, 0) succeeds iff PID is alive and signalable."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # PID exists but is owned by a different user — treat as alive (safe default).
        return True
    except OSError as e:
        if e.errno == errno.ESRCH:
            return False
        return True


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        # Accept both with and without trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


class ConsolidatorLock:
    """Filesystem lock at <worktree>/.diffmem/consolidator.lock.

    Reclaims locks whose PID is dead AND whose started_at is older than 30 min.
    """

    def __init__(self, worktree: Union[str, Path]):
        self.worktree = Path(worktree)
        self.lock_dir = self.worktree / ".diffmem"
        self.lock_path = self.lock_dir / "consolidator.lock"

    def _read_existing(self) -> Optional[dict]:
        if not self.lock_path.exists():
            return None
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("LOCK_READ_FAIL: path=%s err=%s — treating as stale", self.lock_path, e)
            return None

    def _is_stale(self, data: dict) -> bool:
        pid = data.get("pid", -1)
        started_at = data.get("started_at", "")
        if not isinstance(pid, int) or pid <= 0:
            return True
        if _pid_alive(pid):
            return False
        parsed = _parse_iso(started_at)
        if parsed is None:
            # Dead PID + unparseable timestamp → safe to reclaim.
            return True
        now = datetime.now(timezone.utc)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        age = (now - parsed).total_seconds()
        return age >= STALE_AFTER_SECONDS

    def __enter__(self) -> "ConsolidatorLock":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_existing()
        if existing is not None and not self._is_stale(existing):
            raise LockBusyError(
                pid=existing.get("pid", -1),
                started_at=existing.get("started_at", "unknown"),
            )
        if existing is not None:
            logger.info(
                "LOCK_RECLAIMED: stale pid=%s started_at=%s",
                existing.get("pid"),
                existing.get("started_at"),
            )
        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.lock_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.debug("LOCK_ACQUIRED: pid=%s path=%s", payload["pid"], self.lock_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
            logger.debug("LOCK_RELEASED: path=%s", self.lock_path)
        except OSError as e:
            logger.warning("LOCK_RELEASE_FAIL: path=%s err=%s", self.lock_path, e)
