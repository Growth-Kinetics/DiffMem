# CAPABILITY: Dedupe tool implementation.
# INPUTS: ConsolidatorAgent (for LLM client, repo, worktree).
# OUTPUTS: Per-merge `consolidate(dedupe):` commits. Returns result dict.
# CONSTRAINTS: Deterministic prefilter bounds LLM calls. Only same_entity=true
#              AND confidence=high pairs merge. Loser's filename stem → survivor.aliases.

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ._shared import (
    calculate_memory_strength,
    get_file_git_stats,
    rebuild_master_index,
    scan_entities,
    write_with_semantic_index,
)

logger = logging.getLogger(__name__)


# --- prefilter ----------------------------------------------------------------


NAME_SIMILARITY_THRESHOLD = 0.8
MIN_OVERLAP_RELATED = 2
MIN_OVERLAP_HARD_CUES = 3


def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _overlap(a: List[str], b: List[str]) -> int:
    return len(set(map(str.lower, a or [])) & set(map(str.lower, b or [])))


def find_candidate_pairs(entities: List[Dict[str, Any]]) -> List[Tuple[Dict, Dict]]:
    """Pairs of entity dicts that pass the prefilter.

    Rule: same `type`, name similarity ≥ 0.8, AND
    (≥2 overlapping related_entities OR ≥3 overlapping hard_cues OR
     one filename is the other's prefix/contains the other — disambiguator case).
    """
    pairs: List[Tuple[Dict, Dict]] = []
    n = len(entities)
    for i in range(n):
        a = entities[i]
        si_a = a["semantic_index"]
        for j in range(i + 1, n):
            b = entities[j]
            si_b = b["semantic_index"]
            if (si_a.get("type") or "").lower() != (si_b.get("type") or "").lower():
                continue
            name_a = si_a.get("name", "") or a["file"].stem
            name_b = si_b.get("name", "") or b["file"].stem
            sim = _name_similarity(name_a, name_b)
            stem_a = a["file"].stem.lower()
            stem_b = b["file"].stem.lower()
            disambiguator_match = (
                stem_a in stem_b or stem_b in stem_a
            ) and stem_a != stem_b
            if sim < NAME_SIMILARITY_THRESHOLD and not disambiguator_match:
                continue
            rel_overlap = _overlap(
                si_a.get("related_entities", []), si_b.get("related_entities", [])
            )
            cue_overlap = _overlap(si_a.get("hard_cues", []), si_b.get("hard_cues", []))
            if (
                rel_overlap >= MIN_OVERLAP_RELATED
                or cue_overlap >= MIN_OVERLAP_HARD_CUES
                or disambiguator_match
            ):
                pairs.append((a, b))
                logger.info(
                    "DEDUPE_CANDIDATE: a=%s b=%s sim=%.2f rel_overlap=%d cue_overlap=%d disambig=%s",
                    a["path"],
                    b["path"],
                    sim,
                    rel_overlap,
                    cue_overlap,
                    disambiguator_match,
                )
    return pairs


# --- survivor selection -------------------------------------------------------


def pick_survivor_loser(a: Dict[str, Any], b: Dict[str, Any], repo, worktree: Path) -> Tuple[Dict, Dict]:
    """Survivor = higher memory_strength. Tie-break: longer filename
    (disambiguator wins). Final tie-break: lexicographic order."""

    def strength(ent: Dict[str, Any]) -> float:
        si = ent["semantic_index"]
        if "memory_strength" in si:
            try:
                return float(si["memory_strength"])
            except (TypeError, ValueError):
                pass
        if repo is not None:
            stats = get_file_git_stats(repo, worktree, ent["file"])
            return calculate_memory_strength(stats["number_of_edits"], stats["last_update"])
        return 0.0

    sa, sb = strength(a), strength(b)
    if sa > sb:
        return a, b
    if sb > sa:
        return b, a
    # tie: longer filename wins
    if len(a["file"].name) > len(b["file"].name):
        return a, b
    if len(b["file"].name) > len(a["file"].name):
        return b, a
    # final tie: lexicographic
    return (a, b) if a["file"].name < b["file"].name else (b, a)


# --- LLM-driven judge + merge -------------------------------------------------


LLMCall = Callable[[str, str, bool], Any]  # (prompts_dir name, formatted prompt, is_json) -> result


def judge_pair(
    prompts_dir: Path,
    llm_call: Callable[[str, bool], Any],
    survivor: Dict[str, Any],
    loser: Dict[str, Any],
) -> Dict[str, Any]:
    tmpl = (prompts_dir / "dedupe_judge.txt").read_text(encoding="utf-8")
    prompt = tmpl.format(
        file_a_path=survivor["path"],
        file_a_content=survivor["content"],
        file_b_path=loser["path"],
        file_b_content=loser["content"],
    )
    resp = llm_call(prompt, True)
    if not isinstance(resp, dict):
        return {"same_entity": False, "confidence": "low", "rationale": "LLM returned non-dict"}
    return resp


def merge_pair(
    prompts_dir: Path,
    llm_call: Callable[[str, bool], Any],
    survivor: Dict[str, Any],
    loser: Dict[str, Any],
) -> str:
    """Returns merged content (Markdown body + SEMANTIC INDEX). The LLM is
    instructed to include the SEMANTIC INDEX itself; we sanity-check below."""
    si_s = survivor["semantic_index"]
    si_l = loser["semantic_index"]
    tmpl = (prompts_dir / "dedupe_merge.txt").read_text(encoding="utf-8")
    prompt = tmpl.format(
        survivor_path=survivor["path"],
        survivor_strength=si_s.get("memory_strength", "?"),
        survivor_content=survivor["content"],
        loser_path=loser["path"],
        loser_strength=si_l.get("memory_strength", "?"),
        loser_content=loser["content"],
    )
    merged = llm_call(prompt, False)
    if not isinstance(merged, str) or not merged.strip():
        # LLM failed — fall back to a deterministic concatenation that at least
        # preserves both contents and updates aliases.
        logger.warning("DEDUPE_MERGE_LLM_FAIL: falling back to deterministic merge")
        merged = _deterministic_merge(survivor, loser)
    else:
        # Ensure the alias from the loser is present in the SEMANTIC INDEX.
        merged = _ensure_alias(merged, loser["file"].stem)
    return merged


def _deterministic_merge(survivor: Dict[str, Any], loser: Dict[str, Any]) -> str:
    """Fallback if the LLM returns nothing usable. Preserves both bodies,
    rebuilds a sane SEMANTIC INDEX from the union."""
    from ._shared import strip_semantic_index

    body_s = strip_semantic_index(survivor["content"]).rstrip()
    body_l = strip_semantic_index(loser["content"]).rstrip()
    si_s = dict(survivor["semantic_index"])
    si_l = loser["semantic_index"]

    aliases = list(dict.fromkeys((si_s.get("aliases") or []) + (si_l.get("aliases") or []) + [loser["file"].stem]))
    hard_cues = list(dict.fromkeys((si_s.get("hard_cues") or []) + (si_l.get("hard_cues") or [])))
    soft_cues = list(dict.fromkeys((si_s.get("soft_cues") or []) + (si_l.get("soft_cues") or [])))
    emo_cues = list(dict.fromkeys((si_s.get("emotional_cues") or []) + (si_l.get("emotional_cues") or [])))
    related = list(dict.fromkeys((si_s.get("related_entities") or []) + (si_l.get("related_entities") or [])))

    si_s["aliases"] = aliases
    si_s["hard_cues"] = hard_cues
    si_s["soft_cues"] = soft_cues
    si_s["emotional_cues"] = emo_cues
    si_s["related_entities"] = related

    body = body_s + "\n\n## Merged from " + loser["file"].stem + "\n\n" + body_l
    return write_with_semantic_index(body, si_s)


def _ensure_alias(merged: str, loser_stem: str) -> str:
    """If the merged content has a SEMANTIC INDEX block, make sure the loser's
    stem is in `aliases`. If parsing fails or the block is missing, leave as-is
    (caller may log)."""
    from ._shared import extract_semantic_index, write_with_semantic_index, strip_semantic_index

    si = extract_semantic_index(merged)
    if si is None:
        return merged
    aliases = si.get("aliases") or []
    if loser_stem not in aliases:
        aliases = list(aliases) + [loser_stem]
        si["aliases"] = aliases
    return write_with_semantic_index(strip_semantic_index(merged), si)


# --- orchestrator -------------------------------------------------------------


def run(
    *,
    worktree: Path,
    repo,
    prompts_dir: Path,
    llm_call: Callable[[str, bool], Any],
    user_id: str,
) -> Dict[str, Any]:
    """Execute the dedupe pipeline. Returns the canonical result dict."""
    entities = scan_entities(worktree)
    candidates = find_candidate_pairs(entities)
    commits: List[str] = []
    merges = 0

    for a, b in candidates:
        survivor, loser = pick_survivor_loser(a, b, repo, worktree)
        verdict = judge_pair(prompts_dir, llm_call, survivor, loser)
        if not verdict.get("same_entity") or verdict.get("confidence") != "high":
            logger.info(
                "DEDUPE_REJECT: survivor=%s loser=%s verdict=%s",
                survivor["path"],
                loser["path"],
                verdict,
            )
            continue

        merged = merge_pair(prompts_dir, llm_call, survivor, loser)

        # Write survivor + delete loser
        survivor["file"].write_text(merged, encoding="utf-8")
        repo.git.add(str(survivor["file"].relative_to(worktree)))
        repo.git.rm(str(loser["file"].relative_to(worktree)))

        # Commit per merge
        msg = f"consolidate(dedupe): {survivor['file'].stem} ← {loser['file'].stem}"
        repo.index.commit(msg)
        commit_hash = repo.head.commit.hexsha
        commits.append(commit_hash)
        merges += 1
        logger.info("DEDUPE_MERGED: survivor=%s loser=%s commit=%s", survivor["path"], loser["path"], commit_hash[:8])

    # Rebuild master index only if anything happened
    if merges:
        rebuild_master_index(worktree, user_id, repo=repo)
        if repo.is_dirty(untracked_files=True):
            repo.git.add("index.md")
            repo.index.commit("consolidate(dedupe): rebuild master index.md")
            commits.append(repo.head.commit.hexsha)

    return {
        "status": "ok",
        "tool": "dedupe",
        "commits": commits,
        "candidates_evaluated": len(candidates),
        "merges_performed": merges,
        "summary": f"Evaluated {len(candidates)} candidate pair(s); merged {merges}.",
    }
