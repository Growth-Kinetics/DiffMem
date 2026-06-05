# CAPABILITY: Redistribute tool implementation.
# INPUTS: ConsolidatorAgent (worktree, repo, LLM call, prompts dir).
# OUTPUTS: One `consolidate(redistribute):` commit per oversized source entity.
#          Master-index rebuild commit at end.
# CONSTRAINTS: Soft cap default 32k tokens (len // 4 heuristic). Skips the
#              SEMANTIC INDEX section. Prefers smaller-token target entities.

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from ._shared import (
    estimate_tokens,
    extract_semantic_index,
    rebuild_master_index,
    replace_first,
    scan_entities,
    strip_semantic_index,
    write_with_semantic_index,
)

logger = logging.getLogger(__name__)


_SLUG = re.compile(r"[^a-z0-9]+")


def _slugify(name: str) -> str:
    s = _SLUG.sub("_", name.strip().lower()).strip("_")
    return s or "untitled"


def _candidates_block(entities: List[Dict[str, Any]], exclude: Path) -> str:
    """Render the candidate-entities block for the analyze prompt.

    Sorted ascending by tokens so the LLM sees small targets first
    (balancing rule)."""
    rows: List[Dict[str, Any]] = []
    for ent in entities:
        if ent["file"] == exclude:
            continue
        si = ent["semantic_index"]
        rows.append(
            {
                "path": ent["path"],
                "tokens": ent["tokens"],
                "name": si.get("name", ent["file"].stem),
                "type": si.get("type", "unknown"),
                "role": si.get("role", ""),
                "hard_cues": (si.get("hard_cues") or [])[:5],
            }
        )
    rows.sort(key=lambda r: r["tokens"])
    return "\n".join(
        f"- {r['path']} | tokens={r['tokens']} | name={r['name']} | type={r['type']} | role={r['role']} | cues={','.join(r['hard_cues'])}"
        for r in rows
    )


def _user_entity_path(worktree: Path, user_id: str) -> Path:
    return worktree / f"{user_id}.md"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _apply_moves(
    source_content: str,
    moves: List[Dict[str, Any]],
    new_contexts: List[Dict[str, Any]],
    worktree: Path,
    *,
    llm_call: Callable[[str, bool], Any],
    contexts_folder: Path = None,
    prompts_dir: Path,
) -> Dict[str, Any]:
    """Mutate source_content in-memory by removing all source_sections.
    Apply destination writes. Returns dict with counts + list of paths touched."""
    # SEMANTIC INDEX of source is off-limits — split it off.
    si_block_start = source_content.find("## SEMANTIC INDEX")
    if si_block_start == -1:
        body, tail = source_content, ""
    else:
        body, tail = source_content[:si_block_start], source_content[si_block_start:]

    touched_paths: List[Path] = []
    successful_moves = 0
    successful_new = 0

    # 1) attributed moves
    for mv in moves:
        src_section = mv.get("source_section", "")
        target_rel = mv.get("target_entity", "")
        extracted = mv.get("extracted_content", "")
        if not src_section or not target_rel or not extracted:
            continue
        target = (worktree / target_rel).resolve()
        if not target.exists():
            logger.warning("REDISTRIBUTE_TARGET_MISSING: %s", target_rel)
            continue
        new_body, did = replace_first(body, src_section, "")
        if not did:
            logger.warning("REDISTRIBUTE_SOURCE_NOT_FOUND: target=%s", target_rel)
            continue
        body = new_body
        # Append to target above its SEMANTIC INDEX block.
        tgt_content = _read_text(target)
        tgt_si_pos = tgt_content.find("## SEMANTIC INDEX")
        if tgt_si_pos == -1:
            new_tgt = tgt_content.rstrip() + "\n\n" + extracted.strip() + "\n"
        else:
            new_tgt = (
                tgt_content[:tgt_si_pos].rstrip()
                + "\n\n"
                + extracted.strip()
                + "\n\n"
                + tgt_content[tgt_si_pos:]
            )
        _write_text(target, new_tgt)
        touched_paths.append(target)
        successful_moves += 1
        logger.info("REDISTRIBUTE_MOVED: target=%s", target_rel)

    # 2) new contexts
    for nc in new_contexts:
        name = nc.get("name", "")
        extracted = nc.get("extracted_content", "")
        src_section = nc.get("source_section", "")
        if not name or not extracted:
            continue
        slug = _slugify(name)
        # Use contexts_folder from ontology; default to memories/contexts for personal
        _ctx_folder = contexts_folder if contexts_folder is not None else (worktree / "memories" / "contexts")
        new_path = _ctx_folder / f"{slug}.md"
        if new_path.exists():
            logger.warning("REDISTRIBUTE_NEW_CONTEXT_EXISTS: %s — skipping", new_path)
            continue
        # Remove source_section from the source body, if provided.
        if src_section:
            new_body, did = replace_first(body, src_section, "")
            if did:
                body = new_body
        # Build SEMANTIC INDEX for the new context.
        si = _build_semantic_index(
            content=extracted,
            slug=slug,
            llm_call=llm_call,
            prompts_dir=prompts_dir,
            fallback_name=name,
        )
        si["file"] = str(new_path.relative_to(worktree)).replace("\\", "/")
        final_content = write_with_semantic_index(extracted.rstrip() + "\n", si)
        _write_text(new_path, final_content)
        touched_paths.append(new_path)
        successful_new += 1
        logger.info("REDISTRIBUTE_NEW_CONTEXT: path=%s", new_path)

    new_source = body.rstrip() + ("\n\n" + tail if tail else "\n")
    return {
        "new_source_content": new_source,
        "touched_paths": touched_paths,
        "moves_applied": successful_moves,
        "new_contexts_created": successful_new,
    }


def _build_semantic_index(
    *,
    content: str,
    slug: str,
    llm_call: Callable[[str, bool], Any],
    prompts_dir: Path,
    fallback_name: str,
) -> Dict[str, Any]:
    tmpl = (prompts_dir / "build_semantic_index.txt").read_text(encoding="utf-8")
    prompt = tmpl.format(file_content=content[:6000])
    resp = llm_call(prompt, True)
    if isinstance(resp, dict) and resp.get("name"):
        return resp
    # Fallback skeleton.
    return {
        "name": fallback_name,
        "aliases": [],
        "type": "concept",
        "role": "extracted theme",
        "strength": "Low",
        "hard_cues": [],
        "soft_cues": [],
        "emotional_cues": [],
        "related_entities": [],
    }


def _identify_oversized(entities: List[Dict[str, Any]], soft_cap: int) -> List[Dict[str, Any]]:
    oversized = [e for e in entities if e["tokens"] > soft_cap]
    oversized.sort(key=lambda e: e["tokens"], reverse=True)
    return oversized


def _scan_with_user_entity(
    worktree: Path,
    user_id: str,
    entity_dirs: List[Path] = None,
) -> List[Dict[str, Any]]:
    """scan_entities + the user entity at <worktree>/{user_id}.md, since the
    user file lives at the worktree root, not under entity dirs."""
    entities = scan_entities(worktree, entity_dirs=entity_dirs)
    user_file = _user_entity_path(worktree, user_id)
    if user_file.exists():
        try:
            content = _read_text(user_file)
            si = extract_semantic_index(content) or {}
            si["file"] = f"{user_id}.md"
            entities.append(
                {
                    "file": user_file,
                    "path": f"{user_id}.md",
                    "content": content,
                    "semantic_index": si,
                    "tokens": estimate_tokens(content),
                }
            )
        except OSError as e:
            logger.warning("USER_ENTITY_READ_FAIL: %s", e)
    return entities


def run(
    *,
    worktree: Path,
    repo,
    prompts_dir: Path,
    llm_call: Callable[[str, bool], Any],
    user_id: str,
    soft_cap_tokens: int = 32000,
    entity_dirs: List[Path] = None,
    contexts_folder: Path = None,
) -> Dict[str, Any]:
    entities = _scan_with_user_entity(worktree, user_id, entity_dirs=entity_dirs)
    oversized = _identify_oversized(entities, soft_cap_tokens)

    if not oversized:
        return {
            "status": "ok",
            "tool": "redistribute",
            "commits": [],
            "oversized_entities": 0,
            "total_moves": 0,
            "new_contexts": 0,
            "summary": f"No entity exceeded soft cap ({soft_cap_tokens}).",
        }

    commits: List[str] = []
    total_moves = 0
    total_new = 0

    tmpl = (prompts_dir / "redistribute_analyze.txt").read_text(encoding="utf-8")

    for src in oversized:
        candidates_str = _candidates_block(entities, exclude=src["file"])
        prompt = tmpl.format(
            source_path=src["path"],
            source_tokens=src["tokens"],
            soft_cap=soft_cap_tokens,
            source_content=src["content"],
            candidates_block=candidates_str,
        )
        plan = llm_call(prompt, True)
        if not isinstance(plan, dict):
            logger.warning("REDISTRIBUTE_PLAN_INVALID: source=%s", src["path"])
            continue
        moves = plan.get("moves", []) or []
        new_contexts = plan.get("new_contexts", []) or []
        if not moves and not new_contexts:
            logger.info("REDISTRIBUTE_NOOP: source=%s (LLM proposed no moves)", src["path"])
            continue

        result = _apply_moves(
            src["content"],
            moves,
            new_contexts,
            worktree,
            llm_call=llm_call,
            prompts_dir=prompts_dir,
            contexts_folder=contexts_folder,
        )

        # Write the slimmed source.
        _write_text(src["file"], result["new_source_content"])

        # Refresh the in-memory entities list so subsequent oversized
        # candidates see updated token counts.
        for e in entities:
            if e["file"] == src["file"]:
                e["content"] = result["new_source_content"]
                e["tokens"] = estimate_tokens(result["new_source_content"])
        # Targets / new contexts also need refreshing so the next round picks
        # them up at their new size.
        for tp in result["touched_paths"]:
            new_content = _read_text(tp)
            found = False
            for e in entities:
                if e["file"] == tp:
                    e["content"] = new_content
                    e["tokens"] = estimate_tokens(new_content)
                    found = True
                    break
            if not found:
                # newly created context — add to entities for future passes
                rel = str(tp.relative_to(worktree)).replace("\\", "/")
                si = extract_semantic_index(new_content) or {}
                entities.append(
                    {
                        "file": tp,
                        "path": rel,
                        "content": new_content,
                        "semantic_index": si,
                        "tokens": estimate_tokens(new_content),
                    }
                )

        # Stage and commit per source entity.
        rel_paths = [str(src["file"].relative_to(worktree))] + [
            str(tp.relative_to(worktree)) for tp in result["touched_paths"]
        ]
        for rp in rel_paths:
            repo.git.add(rp)
        moves_n = result["moves_applied"]
        new_n = result["new_contexts_created"]
        msg = (
            f"consolidate(redistribute): slim {src['file'].stem} "
            f"({moves_n} moves, {new_n} new contexts)"
        )
        repo.index.commit(msg)
        commits.append(repo.head.commit.hexsha)
        total_moves += moves_n
        total_new += new_n
        logger.info(
            "REDISTRIBUTE_COMMIT: source=%s moves=%d new_contexts=%d commit=%s",
            src["path"],
            moves_n,
            new_n,
            commits[-1][:8],
        )

    if commits:
        rebuild_master_index(worktree, user_id, repo=repo)
        if repo.is_dirty(untracked_files=True):
            repo.git.add("index.md")
            repo.index.commit("consolidate(redistribute): rebuild master index.md")
            commits.append(repo.head.commit.hexsha)

    return {
        "status": "ok",
        "tool": "redistribute",
        "commits": commits,
        "oversized_entities": len(oversized),
        "total_moves": total_moves,
        "new_contexts": total_new,
        "summary": (
            f"Processed {len(oversized)} oversized entity(ies); "
            f"performed {total_moves} attributed move(s) and "
            f"{total_new} new context extraction(s)."
        ),
    }
