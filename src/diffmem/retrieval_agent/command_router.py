"""
Sandboxed command router for the retrieval agent.

Single run(command="...") interface with whitelisted commands,
chain parsing (|, &&, ||, ;), and a two-layer execution/presentation
architecture inspired by the Manus/*nix agent pattern.
"""

import subprocess
import shlex
import time
import logging
import platform
import re
from pathlib import Path
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

IS_WINDOWS = platform.system() == "Windows"

WHITELISTED_COMMANDS = {
    "cat", "head", "tail", "grep", "ls", "wc",
    "git",
}

# On Windows, map Unix commands to git-bash equivalents or git builtins.
# Git for Windows ships with a full Unix toolset at <git>/usr/bin/.
# We prepend the git-bash path so cat, grep, ls, head, tail, wc all work.
_GIT_BASH_BIN: Optional[str] = None

def _get_git_bash_bin() -> Optional[str]:
    """Find the usr/bin directory in Git for Windows installation."""
    global _GIT_BASH_BIN
    if _GIT_BASH_BIN is not None:
        return _GIT_BASH_BIN

    try:
        result = subprocess.run(
            ["git", "--exec-path"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            git_exec = Path(result.stdout.strip())
            usr_bin = git_exec.parent.parent / "usr" / "bin"
            if usr_bin.exists():
                _GIT_BASH_BIN = str(usr_bin)
                return _GIT_BASH_BIN
    except Exception:
        pass

    common_paths = [
        Path("C:/Program Files/Git/usr/bin"),
        Path("C:/Program Files (x86)/Git/usr/bin"),
    ]
    for p in common_paths:
        if p.exists():
            _GIT_BASH_BIN = str(p)
            return _GIT_BASH_BIN

    _GIT_BASH_BIN = ""
    return None

WHITELISTED_GIT_SUBCOMMANDS = {
    "log", "diff", "blame", "show", "rev-list", "shortlog",
}

MAX_OUTPUT_LINES = 150
MAX_OUTPUT_BYTES = 30_000
COMMAND_TIMEOUT_SECONDS = 10


def _is_text(data: bytes) -> bool:
    if b'\x00' in data:
        return False
    try:
        data.decode('utf-8')
    except UnicodeDecodeError:
        return False
    control_chars = sum(1 for b in data if b < 32 and b not in (9, 10, 13))
    if len(data) > 0 and control_chars / len(data) > 0.1:
        return False
    return True


def _validate_command(tokens: List[str]) -> Optional[str]:
    """Validate a single command against the whitelist. Returns error string or None."""
    if not tokens:
        return "[error] empty command"

    base_cmd = Path(tokens[0]).name

    if base_cmd not in WHITELISTED_COMMANDS:
        allowed = ", ".join(sorted(WHITELISTED_COMMANDS))
        return f"[error] unknown command: {base_cmd}. Available: {allowed}"

    if base_cmd == "git":
        if len(tokens) < 2:
            return "[error] git: requires subcommand. Available: log, diff, blame, show, rev-list, shortlog"
        sub = tokens[1]
        if sub not in WHITELISTED_GIT_SUBCOMMANDS:
            allowed = ", ".join(sorted(WHITELISTED_GIT_SUBCOMMANDS))
            return f"[error] git {sub}: not allowed. Available git subcommands: {allowed}"

    return None


def _split_chain(command: str) -> List[Tuple[str, str]]:
    """
    Split a command string into segments by chain operators.
    Returns list of (operator, command_str) tuples.
    The first segment has operator "".
    Pipe segments are grouped together as a single unit.
    """
    segments = []
    current = []
    i = 0
    chars = command

    while i < len(chars):
        c = chars[i]

        if c == '|' and i + 1 < len(chars) and chars[i + 1] == '|':
            segments.append(("||" if segments or current else "", "".join(current).strip()))
            current = []
            i += 2
            continue
        elif c == '&' and i + 1 < len(chars) and chars[i + 1] == '&':
            segments.append(("&&" if segments or current else "", "".join(current).strip()))
            current = []
            i += 2
            continue
        elif c == ';':
            segments.append((";" if segments or current else "", "".join(current).strip()))
            current = []
            i += 1
            continue

        current.append(c)
        i += 1

    if current:
        remainder = "".join(current).strip()
        if remainder:
            segments.append((";" if segments else "", remainder))

    if segments and segments[0][0] == "":
        pass
    elif segments:
        segments[0] = ("", segments[0][1])

    return segments


def _execute_pipeline(pipeline_str: str, cwd: str) -> Tuple[str, str, int]:
    """
    Execute a pipeline (commands joined by |) as a single shell subprocess.
    Returns (stdout, stderr, returncode).
    """
    pipe_parts = pipeline_str.split('|')

    for part in pipe_parts:
        part = part.strip()
        if not part:
            continue
        try:
            tokens = shlex.split(part)
        except ValueError:
            tokens = part.split()

        err = _validate_command(tokens)
        if err:
            return "", err, 1

    env = None
    cmd_str = pipeline_str

    if IS_WINDOWS:
        import os as _os
        git_bin = _get_git_bash_bin()
        if git_bin:
            env = _os.environ.copy()
            env["PATH"] = git_bin + ";" + env.get("PATH", "")
        # cmd.exe doesn't treat single quotes as string delimiters.
        # LLMs default to Unix-style single quotes for git --format args.
        cmd_str = cmd_str.replace("'", '"')

    try:
        kwargs = dict(
            shell=True,
            cwd=cwd,
            capture_output=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        if env:
            kwargs["env"] = env

        result = subprocess.run(cmd_str, **kwargs)
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        return stdout, stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"[error] command timed out after {COMMAND_TIMEOUT_SECONDS}s", 1
    except Exception as e:
        return "", f"[error] execution failed: {e}", 1


def _apply_presentation_layer(stdout: str, stderr: str, returncode: int,
                               elapsed_ms: int) -> str:
    """
    Layer 2: Format output for LLM consumption.
    Truncation, metadata footer, error guidance, binary guard.
    """
    if stdout and not _is_text(stdout.encode('utf-8', errors='replace')):
        return f"[error] binary output detected. Use grep or head to inspect specific parts.\n[exit:{returncode} | {elapsed_ms}ms]"

    output = stdout

    lines = output.split('\n')
    truncated = False
    if len(lines) > MAX_OUTPUT_LINES or len(output.encode('utf-8', errors='replace')) > MAX_OUTPUT_BYTES:
        output = '\n'.join(lines[:MAX_OUTPUT_LINES])
        total_lines = len(lines)
        total_kb = len(stdout.encode('utf-8', errors='replace')) / 1024
        output += f"\n\n--- output truncated ({total_lines} lines, {total_kb:.1f}KB) ---"
        output += "\nExplore with: grep <pattern> or tail/head to navigate"
        truncated = True

    if returncode != 0 and stderr:
        output += f"\n[stderr] {stderr.strip()}"

    output += f"\n[exit:{returncode} | {elapsed_ms}ms]"

    return output


def run(command: str, worktree_path: str) -> str:
    """
    Execute a sandboxed command in the user's worktree.
    Single entry point for the retrieval agent.

    Returns formatted output string ready for LLM consumption.
    """
    command = command.strip()
    if not command:
        return "[error] empty command. Available: cat, grep, ls, head, tail, wc, git log/diff/blame/show\n[exit:1 | 0ms]"

    logger.info(f"COMMAND_ROUTER: {command}")

    segments = _split_chain(command)

    final_output = ""
    last_returncode = 0

    for operator, segment in segments:
        if not segment:
            continue

        if operator == "&&" and last_returncode != 0:
            break
        if operator == "||" and last_returncode == 0:
            continue

        start = time.monotonic()
        stdout, stderr, returncode = _execute_pipeline(segment, worktree_path)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        last_returncode = returncode

        if operator == "" or operator == ";":
            result = _apply_presentation_layer(stdout, stderr, returncode, elapsed_ms)
            final_output = result
        elif operator == "&&":
            result = _apply_presentation_layer(stdout, stderr, returncode, elapsed_ms)
            final_output = result
        elif operator == "||":
            result = _apply_presentation_layer(stdout, stderr, returncode, elapsed_ms)
            final_output = result

    if not final_output:
        final_output = "[exit:0 | 0ms]"

    return final_output
