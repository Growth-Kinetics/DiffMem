"""
Multi-turn retrieval agent that explores a git-based memory repository
via a single run(command="...") tool and outputs a structured retrieval plan.

Uses OpenAI-compatible API (works with OpenRouter, Cerebras, etc).
"""

import json
import time
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from openai import OpenAI

from .command_router import run as execute_command

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class LLMConfig:
    provider: str = "openrouter"
    model: str = "x-ai/grok-4.1-fast"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens_per_turn: int = 1000

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("RETRIEVAL_AGENT_PROVIDER", "openrouter")

        defaults = {
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "model": "x-ai/grok-4.1-fast",
            },
            "cerebras": {
                "base_url": "https://api.cerebras.ai/v1",
                "api_key_env": "CEREBRAS_API_KEY",
                "model": "qwen-3-235b-instruct",
            },
        }

        cfg = defaults.get(provider, defaults["openrouter"])

        return cls(
            provider=provider,
            model=os.getenv("RETRIEVAL_AGENT_MODEL", cfg["model"]),
            base_url=os.getenv("RETRIEVAL_AGENT_BASE_URL", cfg["base_url"]),
            api_key=os.getenv("RETRIEVAL_AGENT_API_KEY", os.getenv(cfg["api_key_env"])),
            temperature=float(os.getenv("RETRIEVAL_AGENT_TEMPERATURE", "0.1")),
            max_tokens_per_turn=int(os.getenv("RETRIEVAL_AGENT_MAX_TOKENS", "1000")),
        )


@dataclass
class ContentPointer:
    type: str
    path: str = ""
    reason: str = ""
    priority: str = "if_budget_allows"
    est_tokens: int = 0
    git_cmd: str = ""
    line_start: int = 0
    line_end: int = 0


@dataclass
class RetrievalPlan:
    pointers: List[ContentPointer] = field(default_factory=list)
    synthesis: str = ""
    entities_identified: List[str] = field(default_factory=list)
    agent_turns: int = 0
    total_elapsed_ms: int = 0


TOOLS = [{
    "type": "function",
    "function": {
        "name": "run",
        "description": "Execute a read-only command in the memory repository. Available: cat, head, tail, grep, ls, wc, git log/diff/blame/show/rev-list",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (supports |, &&, ||, ; chaining)"
                }
            },
            "required": ["command"]
        }
    }
}]


def _build_system_prompt(user_id: str, max_tokens: int,
                          baseline_tokens: int) -> str:
    template = (PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")

    return template.format(
        user_id=user_id,
        baseline_tokens=baseline_tokens,
        remaining_budget=max_tokens,
    )


def _build_user_message(conversation: List[Dict[str, str]]) -> str:
    formatted = []
    for msg in conversation:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    transcript = "\n".join(formatted)
    return (
        f"Here is the current conversation. Find relevant memory context for it.\n\n"
        f"--- CONVERSATION ---\n{transcript}\n--- END CONVERSATION ---\n\n"
        f"Begin exploring the repository. Start with step 1 of the protocol."
    )


def _parse_retrieval_plan(text: str) -> RetrievalPlan:
    """Parse the agent's final JSON output into a RetrievalPlan."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        cleaned = "\n".join(lines)
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                logger.warning(f"RETRIEVAL_PARSE_FAIL: Could not parse agent output")
                return RetrievalPlan(synthesis=f"Parse failed. Raw: {text[:500]}")
        else:
            logger.warning(f"RETRIEVAL_PARSE_FAIL: No JSON found in agent output")
            return RetrievalPlan(synthesis=f"No JSON found. Raw: {text[:500]}")

    pointers = []
    for p in data.get("pointers", []):
        pointers.append(ContentPointer(
            type=p.get("type", "file"),
            path=p.get("path", ""),
            reason=p.get("reason", ""),
            priority=p.get("priority", "if_budget_allows"),
            est_tokens=p.get("est_tokens", 0),
            git_cmd=p.get("git_cmd", ""),
            line_start=p.get("line_start", 0),
            line_end=p.get("line_end", 0),
        ))

    return RetrievalPlan(
        pointers=pointers,
        synthesis=data.get("synthesis", ""),
        entities_identified=data.get("entities_identified", []),
    )


def run_retrieval_agent(
    worktree_path: str,
    user_id: str,
    conversation: List[Dict[str, str]],
    max_tokens: int = 20000,
    baseline_tokens: int = 5000,
    max_turns: int = 4,
    timeout_seconds: int = 30,
    llm_config: Optional[LLMConfig] = None,
) -> RetrievalPlan:
    """
    Run the multi-turn retrieval agent against a user's worktree.

    Returns a RetrievalPlan with pointers to content the resolver should load.
    """
    if llm_config is None:
        llm_config = LLMConfig.from_env()

    client = OpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )

    system_prompt = _build_system_prompt(user_id, max_tokens, baseline_tokens)
    user_message = _build_user_message(conversation)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    total_start = time.monotonic()
    turns_used = 0

    logger.info(f"RETRIEVAL_AGENT_START: user={user_id} model={llm_config.model} max_turns={max_turns}")

    for turn in range(max_turns):
        elapsed = time.monotonic() - total_start
        if elapsed > timeout_seconds:
            logger.warning(f"RETRIEVAL_AGENT_TIMEOUT: {elapsed:.1f}s > {timeout_seconds}s after {turn} turns")
            break

        turns_used += 1

        try:
            response = client.chat.completions.create(
                model=llm_config.model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens_per_turn,
            )
        except Exception as e:
            logger.error(f"RETRIEVAL_AGENT_LLM_ERROR: {e}")
            break

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            messages.append(message.model_dump())

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {"command": tool_call.function.arguments}

                if fn_name == "run":
                    cmd = args.get("command", "")
                    logger.info(f"RETRIEVAL_AGENT_CMD[{turn}]: {cmd}")
                    result = execute_command(cmd, worktree_path)
                else:
                    result = f"[error] unknown tool: {fn_name}. Use run(command='...')"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            content = message.content or ""
            logger.info(f"RETRIEVAL_AGENT_DONE: turn={turn} output_len={len(content)}")
            plan = _parse_retrieval_plan(content)
            plan.agent_turns = turns_used
            plan.total_elapsed_ms = int((time.monotonic() - total_start) * 1000)
            return plan

    total_ms = int((time.monotonic() - total_start) * 1000)
    logger.warning(f"RETRIEVAL_AGENT_EXHAUSTED: used {turns_used} turns in {total_ms}ms without final output")

    last_content = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            last_content = msg.get("content", "")
            if last_content:
                break
        elif hasattr(msg, "content") and msg.content:
            last_content = msg.content
            break

    if last_content:
        plan = _parse_retrieval_plan(last_content)
    else:
        plan = RetrievalPlan(synthesis="Agent exhausted turns without producing a retrieval plan.")

    plan.agent_turns = turns_used
    plan.total_elapsed_ms = total_ms
    return plan
