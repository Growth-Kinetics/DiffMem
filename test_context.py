"""
A/B comparison of old (v1) vs new (v2 agent) context retrieval.

Reads a conversation from user_test.json, runs both retrieval paths,
and saves outputs + the v2 agent's full conversation trace.

Usage:
    python test_context.py

Outputs saved to test_output/ directory.
"""

import sys
import os
import json
import time
import types
import importlib.util
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Bootstrap: load retrieval_agent modules directly for standalone testing
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "src")

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pkg = types.ModuleType("diffmem")
pkg.__path__ = [os.path.join(SRC, "diffmem")]
sys.modules["diffmem"] = pkg

ra_pkg = types.ModuleType("diffmem.retrieval_agent")
ra_pkg.__path__ = [os.path.join(SRC, "diffmem", "retrieval_agent")]
sys.modules["diffmem.retrieval_agent"] = ra_pkg

cmd_mod = _load_module(
    "diffmem.retrieval_agent.command_router",
    os.path.join(SRC, "diffmem", "retrieval_agent", "command_router.py"),
)
agent_mod = _load_module(
    "diffmem.retrieval_agent.agent",
    os.path.join(SRC, "diffmem", "retrieval_agent", "agent.py"),
)
base_mod = _load_module(
    "diffmem.retrieval_agent.baseline",
    os.path.join(SRC, "diffmem", "retrieval_agent", "baseline.py"),
)
resolver_mod = _load_module(
    "diffmem.retrieval_agent.resolver",
    os.path.join(SRC, "diffmem", "retrieval_agent", "resolver.py"),
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKTREE = r"C:\Users\alexm\Documents\memory_v2\Diffmem-Worktrees\34634821429"
USER_ID = "34634821429"
CONVERSATION_FILE = os.path.join(BASE, "user_test.json")
OUTPUT_DIR = os.path.join(BASE, "test_output")
MAX_TOKENS = 15000  # agent's additional context budget (on top of baseline)
MAX_TURNS = 6

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_json(data, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {path}")


def format_agent_trace(messages):
    """Convert the raw message list into a human-readable trace."""
    lines = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            if role == "system":
                lines.append(f"--- [{i}] SYSTEM PROMPT ({len(msg.get('content',''))} chars) ---")
                lines.append(msg.get("content", "")[:500] + "...")
                lines.append("")
            elif role == "user":
                lines.append(f"--- [{i}] USER MESSAGE ---")
                content = msg.get("content", "")
                lines.append(content[:800] + ("..." if len(content) > 800 else ""))
                lines.append("")
            elif role == "assistant":
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    lines.append(f"--- [{i}] ASSISTANT (tool calls) ---")
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        lines.append(f"  CALL: {fn.get('name', '?')}({fn.get('arguments', '')})")
                elif content:
                    lines.append(f"--- [{i}] ASSISTANT (final output) ---")
                    lines.append(content)
                lines.append("")
            elif role == "tool":
                tool_id = msg.get("tool_call_id", "?")
                content = msg.get("content", "")
                lines.append(f"--- [{i}] TOOL RESULT (id={tool_id[:12]}...) ---")
                if len(content) > 1500:
                    lines.append(content[:1500] + f"\n... ({len(content)} chars total)")
                else:
                    lines.append(content)
                lines.append("")
        else:
            # OpenAI message object (from model_dump)
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", "") or ""
            lines.append(f"--- [{i}] {role.upper()} (object) ---")
            lines.append(content[:500] if content else "(no content)")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Modified agent runner that captures the full message trace
# ---------------------------------------------------------------------------
def run_agent_with_trace(worktree_path, user_id, conversation, max_tokens, baseline_tokens, max_turns):
    """Run the retrieval agent and return (plan, messages_trace)."""
    from openai import OpenAI

    llm_config = agent_mod.LLMConfig.from_env()

    client = OpenAI(
        base_url=llm_config.base_url,
        api_key=llm_config.api_key,
    )

    system_prompt = agent_mod._build_system_prompt(user_id, max_tokens, baseline_tokens)
    user_message = agent_mod._build_user_message(conversation)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    total_start = time.monotonic()
    turns_used = 0

    print(f"\n  Agent starting: model={llm_config.model} max_turns={max_turns}")

    for turn in range(max_turns):
        elapsed = time.monotonic() - total_start
        turns_used += 1

        print(f"  Turn {turn + 1}/{max_turns} ({elapsed:.1f}s elapsed)...", end=" ", flush=True)

        try:
            response = client.chat.completions.create(
                model=llm_config.model,
                messages=messages,
                tools=agent_mod.TOOLS,
                tool_choice="auto",
                temperature=llm_config.temperature,
                max_tokens=2000,
            )
        except Exception as e:
            print(f"LLM ERROR: {e}")
            break

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            messages.append(message.model_dump())

            for tc in message.tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"command": tc.function.arguments}

                cmd = args.get("command", "")
                print(f"run(\"{cmd}\")")

                result = cmd_mod.run(cmd, worktree_path)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            content = message.content or ""
            print(f"DONE ({len(content)} chars)")
            messages.append({"role": "assistant", "content": content})

            plan = agent_mod._parse_retrieval_plan(content)
            plan.agent_turns = turns_used
            plan.total_elapsed_ms = int((time.monotonic() - total_start) * 1000)
            return plan, messages

    total_ms = int((time.monotonic() - total_start) * 1000)
    plan = agent_mod.RetrievalPlan(
        synthesis="Agent exhausted turns without final output.",
        agent_turns=turns_used,
        total_elapsed_ms=total_ms,
    )
    return plan, messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DiffMem Context Retrieval: A/B Comparison")
    print("=" * 70)

    # Load conversation
    with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
        conversation = json.load(f)
    print(f"\nConversation loaded: {len(conversation)} messages from {CONVERSATION_FILE}")
    print(f"Worktree: {WORKTREE}")
    print(f"User ID: {USER_ID}")

    # ------------------------------------------------------------------
    # V2: New agent-based retrieval
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("V2: Git-Native Agent Retrieval")
    print("=" * 70)

    # Phase 0: Baseline (user entity only)
    print("\n[Phase 0] Loading deterministic baseline (user entity only)...")
    t0 = time.monotonic()
    baseline = base_mod.load_baseline(WORKTREE, USER_ID)
    baseline_ms = int((time.monotonic() - t0) * 1000)
    print(f"  User entity: {baseline['user_entity']['tokens']} tokens")
    print(f"  Timeline: {len(baseline['timeline'])} files, {sum(t['tokens'] for t in baseline['timeline'])} tokens")
    print(f"  Total baseline: {baseline['total_tokens']} tokens ({baseline_ms}ms)")

    # Phase 1: Agent
    print("\n[Phase 1] Running retrieval agent...")
    plan, agent_messages = run_agent_with_trace(
        WORKTREE, USER_ID, conversation, MAX_TOKENS,
        baseline["total_tokens"], MAX_TURNS,
    )
    print(f"\n  Agent completed: {plan.agent_turns} turns, {plan.total_elapsed_ms}ms")
    print(f"  Synthesis: {plan.synthesis[:200]}...")
    print(f"  Entities: {plan.entities_identified}")
    print(f"  Pointers: {len(plan.pointers)}")
    for p in plan.pointers:
        print(f"    [{p.priority}] {p.type}: {p.path or p.git_cmd} ({p.est_tokens} est tok) - {p.reason[:80]}")

    # Phase 2: Resolve pointers
    print("\n[Phase 2] Resolving pointers...")
    t0 = time.monotonic()
    agent_blocks = resolver_mod.resolve_pointers(plan, WORKTREE, token_budget=MAX_TOKENS)
    resolve_ms = int((time.monotonic() - t0) * 1000)
    agent_tokens = sum(b['tokens'] for b in agent_blocks)
    print(f"  Resolved {len(agent_blocks)} blocks, {agent_tokens} tokens ({resolve_ms}ms)")
    for b in agent_blocks:
        print(f"    - {b['source']}: {b['tokens']} tok ({b['type']})")

    # Phase 3: ALWAYS_LOAD for identified entities (tail-end safety net)
    print("\n[Phase 3] Loading ALWAYS_LOAD blocks for identified entities...")
    t0 = time.monotonic()
    al_budget = max(1000, MAX_TOKENS - agent_tokens)
    always_load = base_mod.load_always_load_for_entities(WORKTREE, plan.entities_identified, max_tokens=al_budget)
    al_ms = int((time.monotonic() - t0) * 1000)
    al_tokens = sum(b['tokens'] for b in always_load)
    print(f"  Loaded {len(always_load)} blocks for entities {plan.entities_identified}")
    print(f"  ALWAYS_LOAD: {al_tokens} tokens ({al_ms}ms)")
    for b in always_load:
        header = b.get('header', '')[:50].encode('ascii', 'replace').decode()
        print(f"    - {b['source']} [{header}]: {b['tokens']} tok")

    # Assemble v2 output
    v2_result = {
        "user_entity": baseline["user_entity"],
        "recent_timeline": baseline["timeline"],
        "agent_context": agent_blocks,
        "always_load_blocks": always_load,
        "retrieval_plan": {
            "synthesis": plan.synthesis,
            "entities_identified": plan.entities_identified,
            "pointers": [
                {"type": p.type, "path": p.path, "git_cmd": p.git_cmd,
                 "reason": p.reason, "priority": p.priority, "est_tokens": p.est_tokens}
                for p in plan.pointers
            ],
            "agent_turns": plan.agent_turns,
            "agent_elapsed_ms": plan.total_elapsed_ms,
        },
        "session_metadata": {
            "user_id": USER_ID,
            "retrieval_version": "v2_agent",
            "max_tokens": MAX_TOKENS,
            "baseline_tokens": baseline["total_tokens"],
            "agent_tokens": agent_tokens,
            "always_load_tokens": al_tokens,
            "total_tokens": baseline["total_tokens"] + agent_tokens + al_tokens,
            "baseline_ms": baseline_ms,
            "agent_ms": plan.total_elapsed_ms,
            "resolve_ms": resolve_ms,
            "always_load_ms": al_ms,
        },
    }

    # Save outputs
    print("\n[Saving outputs]")
    save_json(v2_result, "v2_context_result.json")
    save_json(conversation, "conversation_input.json")

    # Save the agent trace as both JSON (for programmatic use) and text (for reading)
    serializable_messages = []
    for msg in agent_messages:
        if isinstance(msg, dict):
            serializable_messages.append(msg)
        else:
            serializable_messages.append({"role": str(getattr(msg, "role", "?")), "content": str(getattr(msg, "content", ""))})
    save_json(serializable_messages, "v2_agent_trace.json")

    trace_text = format_agent_trace(agent_messages)
    trace_path = os.path.join(OUTPUT_DIR, "v2_agent_trace.txt")
    with open(trace_path, "w", encoding="utf-8") as f:
        f.write(trace_text)
    print(f"  Saved: {trace_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_v2 = baseline["total_tokens"] + agent_tokens + al_tokens
    total_ms = baseline_ms + plan.total_elapsed_ms + resolve_ms + al_ms
    print(f"\n  V2 (agent):")
    print(f"    Baseline (user entity): {baseline['total_tokens']} tokens ({baseline_ms}ms)")
    print(f"    Agent context:          {agent_tokens} tokens ({plan.total_elapsed_ms}ms, {plan.agent_turns} turns)")
    print(f"    ALWAYS_LOAD (tail):     {al_tokens} tokens for {len(plan.entities_identified)} entities ({al_ms}ms)")
    print(f"    Resolve:                {resolve_ms}ms")
    print(f"    TOTAL:                  {total_v2} tokens, {total_ms}ms")
    print(f"\n  Context breakdown:")
    for b in agent_blocks:
        print(f"    [{b['type']}] {b['source']}: {b['tokens']} tok - {b.get('reason','')[:60]}")
    print(f"\n  Outputs in: {OUTPUT_DIR}/")
    print(f"    v2_context_result.json  - Full context output")
    print(f"    v2_agent_trace.json     - Agent conversation (machine-readable)")
    print(f"    v2_agent_trace.txt      - Agent conversation (human-readable)")


if __name__ == "__main__":
    main()
