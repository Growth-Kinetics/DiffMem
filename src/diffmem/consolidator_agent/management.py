# CAPABILITY: Management agent — integrity-preserving entity mutations for the
#              memory admin surface (ChatBarry /settings/memory).
# INPUTS: Worktree path, user_id, OpenRouter key/model (or llm_call override),
#         ontology profile. Per-op request dicts (paths, types, notes).
# OUTPUTS: `manage(...)`-prefixed git commits + rebuilt master index. Same-type
#          merge enforcement, path sandboxing, User Context provenance.
# CONSTRAINTS: All mutations under ConsolidatorLock. LLM only in merge + add-note
#              (user-forced merge SKIPS the judge — the user IS the judge).
#              Never touches the user entity, index.md, timeline/, sessions/.

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import git
from openai import OpenAI

from .lock import ConsolidatorLock
from . import _dedupe, _shared
from ..frontmatter import (
    merge_frontmatter,
    parse_frontmatter,
    strip_legacy_semantic_index,
    normalize_semantic_index,
)
from ..ontology.loader import OntologyProfile, load_ontology

logger = logging.getLogger(__name__)

LLMCall = Callable[[str, bool], Any]

USER_CONTEXT_HEADER = "## User Context"


# ── path sandboxing ──────────────────────────────────────────────────────────


class ManagementError(ValueError):
    """A user-facing validation failure (maps to HTTP 400 at the route layer)."""


def _safe_rel(worktree: Path, rel: str) -> Path:
    """Validate + resolve a worktree-relative entity path.

    Rejects: absolute paths, traversal, non-.md, missing files, index.md,
    the root user entity, timeline/ and sessions/ trees, repo_guide.md.
    Returns the absolute Path inside the worktree.
    """
    if not rel or not isinstance(rel, str):
        raise ManagementError("path must be a non-empty string")
    if rel.startswith("/") or ".." in Path(rel).parts or Path(rel).is_absolute():
        raise ManagementError(f"path must be worktree-relative: {rel!r}")
    p = Path(rel)
    if p.suffix != ".md":
        raise ManagementError(f"path must be a .md file: {rel!r}")
    top = p.parts[0] if p.parts else ""
    if top in {"timeline", "sessions"} or p.name in {"index.md", "repo_guide.md", "episodes_index.md"}:
        raise ManagementError(f"path is not an entity file: {rel!r}")
    abs_path = (worktree / rel).resolve()
    if not abs_path.exists():
        raise ManagementError(f"entity file not found: {rel!r}")
    return abs_path


def _entity_type_of(worktree: Path, abs_path: Path, ontology: OntologyProfile) -> Optional[str]:
    """Ontology entity-type NAME whose folder contains this path (None if outside).
    Both sides are resolved — macOS temp dirs hand out /var/... while
    Path.resolve() yields /private/var/..., and unresolved comparison breaks
    containment checks."""
    resolved = abs_path.resolve()
    for et in ontology.entity_types:
        folder = (worktree / et["folder"]).resolve()
        try:
            resolved.relative_to(folder)
            return et["name"]
        except ValueError:
            continue
    return None


def _require_entity(worktree: Path, rel: str, ontology: OntologyProfile) -> tuple[Path, str]:
    abs_path = _safe_rel(worktree, rel)
    etype = _entity_type_of(worktree, abs_path, ontology)
    if etype is None:
        # The root user entity or a stray file outside entity dirs.
        raise ManagementError(f"path is not an entity file under the ontology: {rel!r}")
    return abs_path, etype


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not s:
        raise ManagementError("name must contain alphanumeric characters")
    return s


# ── User Context provenance ──────────────────────────────────────────────────


def _append_user_context(content: str, note: str, op: str) -> str:
    """Record a dated bullet under `## User Context` (section created on first
    use). The writer agent reads the full file body on every future update, so
    the note steers later reprocessing. Inserted AFTER frontmatter, BEFORE any
    legacy SEMANTIC INDEX block."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    bullet = f"- **{stamp}** ({op}): {note.strip()}"
    if USER_CONTEXT_HEADER in content:
        head, _, tail = content.partition(USER_CONTEXT_HEADER)
        rest_lines = tail.split("\n")
        # Insert after the last consecutive bullet under the section.
        idx = 0
        for i, line in enumerate(rest_lines[1:], start=1):
            if line.strip().startswith("- ") or not line.strip():
                idx = i
            else:
                break
        rest_lines.insert(idx + 1, bullet)
        content = head + USER_CONTEXT_HEADER + "\n".join(rest_lines)
    else:
        section = f"\n{USER_CONTEXT_HEADER}\n{bullet}\n"
        if "## SEMANTIC INDEX" in content:
            before, _, after = content.partition("## SEMANTIC INDEX")
            content = before.rstrip("\n") + "\n" + section + "\n## SEMANTIC INDEX" + after
        else:
            content = content.rstrip("\n") + "\n" + section
    return content


def _si_type_for(ontology: OntologyProfile, etype_name: str) -> Optional[str]:
    for et in ontology.entity_types:
        if et["name"] == etype_name:
            return et["index_type"]
    return None


# ── agent ────────────────────────────────────────────────────────────────────


class ManagementAgent:
    """Entity-management operations mirroring ConsolidatorAgent's constructor
    shape (same DI surface: inject `llm_call` for tests)."""

    def __init__(
        self,
        repo_path: str,
        user_id: str,
        openrouter_api_key: str,
        model: Optional[str] = None,
        llm_call: Optional[LLMCall] = None,
        validate_paths: bool = True,
        ontology: Optional[OntologyProfile] = None,
    ) -> None:
        if not model:
            raise ValueError("model must be set via argument or DEFAULT_MODEL env var")
        # Resolve once: _safe_rel() resolves the paths it validates, and
        # relativizing a resolved path against an unresolved worktree root
        # breaks on symlinked roots (macOS /tmp vs /private/tmp).
        self.repo_path = Path(repo_path).resolve()
        self.user_id = user_id
        self.model = model
        self.prompts_path = Path(__file__).parent / "prompts"
        self.ontology: OntologyProfile = ontology if ontology is not None else load_ontology()
        if validate_paths and not self.repo_path.exists():
            raise FileNotFoundError(f"Worktree not found: {self.repo_path}")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        self._llm_call_override = llm_call
        self.logger = logger
        self.logger.info("MANAGEMENT_INIT: repo=%s user=%s", self.repo_path, self.user_id)

    # --- plumbing -------------------------------------------------------------

    def _call_llm(self, prompt: str, is_json: bool = True) -> Any:
        if self._llm_call_override is not None:
            return self._llm_call_override(prompt, is_json)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.15,
                response_format={"type": "json_object"} if is_json else None,
            )
            content = resp.choices[0].message.content
            if is_json:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {}
            return content
        except Exception as e:
            self.logger.error("MANAGEMENT_LLM_FAIL: err=%s", e)
            return {} if is_json else ""

    def _repo(self) -> git.Repo:
        return git.Repo(self.repo_path)

    def _lock(self) -> ConsolidatorLock:
        return ConsolidatorLock(self.repo_path)

    def _load_entity(self, abs_path: Path) -> Dict[str, Any]:
        """Load an entity dict in the scan_entities shape (path/content/SI)."""
        content = abs_path.read_text(encoding="utf-8")
        si = _shared.extract_semantic_index(content) or {}
        return {
            "file": abs_path,
            "path": str(abs_path.relative_to(self.repo_path)).replace("\\", "/"),
            "content": content,
            "semantic_index": si,
        }

    def _commit(self, repo: git.Repo, message: str) -> str:
        repo.index.commit(message)
        return repo.head.commit.hexsha

    def _rebuild_index(self, repo: git.Repo) -> Optional[str]:
        # Management ops rebuild WITHOUT per-file git stats (repo=None) — the
        # shared rebuild_master_index calls `git log` + `git rev-list` per
        # entity file, which is ~2 subprocesses × 1000+ files = 30-120s on a
        # large store. Management ops (merge/move/rename/etc.) need a FAST
        # index refresh (the user is waiting at the UI); the next normal chat
        # ingest's _rebuild_master_index (writer path) refreshes git stats.
        _shared.rebuild_master_index(
            self.repo_path, self.user_id, repo=None,
            entity_dirs=self.ontology.entity_dirs(self.repo_path),
        )
        if repo.is_dirty(untracked_files=True):
            repo.git.add("index.md")
            return self._commit(repo, "manage: rebuild master index.md")
        return None

    def _entity_dirs_paths(self) -> List[Path]:
        return self.ontology.entity_dirs(self.repo_path)

    # ── MERGE (LLM; user-forced — no judge) ─────────────────────────────────

    def manage_merge(
        self,
        survivor_path: str,
        loser_paths: List[str],
        strategy: str = "llm",
        context: Optional[str] = None,
        dry_run: bool = False,
        reviewed_markdown: Optional[str] = None,
        reviewed_semantic_index: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge every loser INTO the survivor. Same-type enforced. Loser name
        variants (stem + SI name + aliases) become survivor aliases so old
        spellings never re-create the loser. dry_run returns previews, no commit."""
        with self._lock():
            repo = self._repo()
            surv_abs, surv_type = _require_entity(self.repo_path, survivor_path, self.ontology)
            losers: List[tuple[Path, str, str]] = []
            seen = {survivor_path}
            for lp in loser_paths:
                if lp in seen:
                    continue
                seen.add(lp)
                l_abs, l_type = _require_entity(self.repo_path, lp, self.ontology)
                if l_type != surv_type:
                    raise ManagementError(
                        f"cross-type merge rejected: survivor is '{surv_type}' but "
                        f"'{lp}' is '{l_type}'. Move entities to the same type first."
                    )
                losers.append((l_abs, lp, l_type))

            if not losers:
                raise ManagementError("no loser entities to merge (all identical to survivor?)")

            # ── REVIEWED COMMIT ──────────────────────────────────────────────
            # The client already ran dry_run=true (one LLM merge) and the user
            # reviewed/edited the returned markdown. Commit THAT body verbatim —
            # no second LLM call. Loser aliases are still forced in (otherwise
            # the writer re-creates the loser on old spellings), and loser cues
            # are unioned into the SEMANTIC INDEX so nothing is dropped even if
            # the user's edit trimmed the body.
            if reviewed_markdown and reviewed_markdown.strip() and not dry_run:
                survivor = self._load_entity(surv_abs)
                si = dict(reviewed_semantic_index or survivor["semantic_index"] or {})
                merged_aliases: List[str] = []
                for key in ("aliases", "hard_cues", "soft_cues", "emotional_cues", "related_entities"):
                    si[key] = list(si.get(key) or [])
                for l_abs, _, _ in losers:
                    loser = self._load_entity(l_abs)
                    lsi = loser["semantic_index"] or {}
                    for key in ("hard_cues", "soft_cues", "emotional_cues", "related_entities"):
                        si[key] = list(dict.fromkeys((si.get(key) or []) + (lsi.get(key) or [])))
                    merged_aliases.extend(_dedupe.loser_name_variants(loser))
                si["aliases"] = list(dict.fromkeys((si.get("aliases") or []) + merged_aliases))

                body = strip_legacy_semantic_index(reviewed_markdown).strip() + "\n"
                full = _shared.write_with_semantic_index(body, normalize_semantic_index(si))
                full = _dedupe._ensure_aliases(full, sorted(set(merged_aliases)))
                if context:
                    full = _append_user_context(full, context, "merge")

                surv_abs.write_text(full, encoding="utf-8")
                for l_abs, _, _ in losers:
                    repo.git.rm(str(l_abs.relative_to(self.repo_path)))
                repo.git.add(str(surv_abs.relative_to(self.repo_path)))
                commit = self._commit(
                    repo,
                    f"manage(merge): {surv_abs.stem} ← {', '.join(a.stem for a, _, _ in losers)} (reviewed)",
                )
                index_commit = self._rebuild_index(repo)
                self.logger.info(
                    "MANAGE_MERGE_REVIEWED: survivor=%s losers=%d llm=skipped",
                    survivor_path, len(losers),
                )
                return {
                    "status": "ok",
                    "tool": "merge",
                    "commits": [commit] + ([index_commit] if index_commit else []),
                    "survivor_path": survivor_path,
                    "losers_merged": [lp for _, lp, _ in losers],
                    "aliases_added": sorted(set(merged_aliases)),
                    "reviewed": True,
                    "summary": f"Merged {len(losers)} entit(y/ies) into {surv_abs.stem} (user-reviewed).",
                }

            survivor = self._load_entity(surv_abs)
            previews: List[Dict[str, Any]] = []
            merged_content = survivor["content"]
            merged_aliases: List[str] = []

            for l_abs, lp, _ in losers:
                loser = self._load_entity(l_abs)
                survivor["content"] = merged_content
                survivor["semantic_index"] = _shared.extract_semantic_index(merged_content) or {}

                if strategy == "deterministic":
                    merged_content = _dedupe._deterministic_merge(survivor, loser)
                else:
                    merged_content = _dedupe.merge_pair(
                        self.prompts_path, self._call_llm, survivor, loser
                    )
                variants = _dedupe.loser_name_variants(loser)
                merged_aliases.extend(variants)
                merged_content = _dedupe._ensure_aliases(merged_content, variants)
                previews.append({
                    "loser_path": lp,
                    "loser_variants": variants,
                    "preview_markdown": merged_content if dry_run else None,
                })
                self.logger.info(
                    "MANAGE_MERGE_STEP: survivor=%s loser=%s strategy=%s",
                    survivor_path, lp, strategy,
                )

            if context:
                merged_content = _append_user_context(merged_content, context, "merge")

            if dry_run:
                return {
                    "status": "preview",
                    "tool": "merge",
                    "survivor_path": survivor_path,
                    "dry_run": True,
                    "previews": previews,
                    "final_markdown": strip_legacy_semantic_index(merged_content),
                    # Round-tripped by the client on the reviewed commit so the
                    # curated SI survives without a second LLM call.
                    "semantic_index": _shared.extract_semantic_index(merged_content) or {},
                }

            surv_abs.write_text(
                _shared.write_with_semantic_index(
                    strip_legacy_semantic_index(merged_content),
                    normalize_semantic_index(_shared.extract_semantic_index(merged_content) or {}),
                ),
                encoding="utf-8",
            )
            for l_abs, lp, _ in losers:
                repo.git.rm(str(l_abs.relative_to(self.repo_path)))
            repo.git.add(str(surv_abs.relative_to(self.repo_path)))
            commit = self._commit(
                repo,
                f"manage(merge): {surv_abs.stem} ← {', '.join(a.stem for a, _, _ in losers)}",
            )
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "merge",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "survivor_path": survivor_path,
                "losers_merged": [lp for _, lp, _ in losers],
                "aliases_added": sorted(set(merged_aliases)),
                "summary": f"Merged {len(losers)} entit(y/ies) into {surv_abs.stem}.",
            }

    # ── MOVE (sync; type reassignment) ─────────────────────────────────────

    def manage_move(
        self, paths: List[str], to_type: str, context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """git mv entities into another ontology type's folder + rewrite the SI
        `type` field (index_type vocab) in frontmatter and legacy blocks."""
        if to_type not in self.ontology.folder_map:
            raise ManagementError(
                f"unknown entity type '{to_type}'. Available: "
                f"{sorted(self.ontology.folder_map)}"
            )
        with self._lock():
            repo = self._repo()
            dest_folder = self.repo_path / self.ontology.folder_map[to_type]
            dest_folder.mkdir(parents=True, exist_ok=True)
            si_type = _si_type_for(self.ontology, to_type)
            moved: List[str] = []

            for p in paths:
                abs_path, current_type = _require_entity(self.repo_path, p, self.ontology)
                if current_type == to_type:
                    continue  # idempotent no-op
                content = abs_path.read_text(encoding="utf-8")
                si = _shared.extract_semantic_index(content) or {}
                if si_type:
                    si["type"] = si_type
                content = _shared.write_with_semantic_index(
                    strip_legacy_semantic_index(content), si
                )
                if context:
                    content = _append_user_context(content, context, f"moved {current_type} → {to_type}")
                abs_path.write_text(content, encoding="utf-8")

                new_rel = Path(self.ontology.folder_map[to_type]) / abs_path.name
                if (self.repo_path / new_rel).exists():
                    raise ManagementError(f"destination exists: {new_rel}")
                repo.git.mv(str(abs_path.relative_to(self.repo_path)), str(new_rel))
                moved.append(f"{p} → {new_rel}")
                self.logger.info(
                    "MANAGE_MOVE: %s → %s (%s → %s)", p, new_rel, current_type, to_type
                )

            if not moved:
                return {
                    "status": "ok", "tool": "move", "commits": [], "moved": [],
                    "summary": "Nothing to move (already at target type).",
                }
            commit = self._commit(repo, f"manage(move): {len(moved)} → {to_type}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "move",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "moved": moved,
                "summary": f"Moved {len(moved)} entit(y/ies) to {to_type}.",
            }

    # ── RENAME (sync) ────────────────────────────────────────────────────────

    def manage_rename(
        self, path: str, new_name: str, context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Rename an entity: git mv + slugified stem + SI name update + H1
        display-name update; the OLD stem becomes an alias (never re-created)."""
        new_stem = _slugify(new_name)
        with self._lock():
            repo = self._repo()
            abs_path, _ = _require_entity(self.repo_path, path, self.ontology)
            old_stem = abs_path.stem
            if new_stem == old_stem:
                raise ManagementError(f"entity is already named '{old_stem}'")
            rel = abs_path.relative_to(self.repo_path)
            new_rel = rel.parent / f"{new_stem}.md"
            if (self.repo_path / new_rel).exists():
                raise ManagementError(f"destination exists: {new_rel}")

            content = abs_path.read_text(encoding="utf-8")
            si = _shared.extract_semantic_index(content) or {}
            old_display = str(si.get("name") or old_stem.replace("_", " "))
            si["name"] = new_stem
            aliases = list(si.get("aliases") or [])
            for v in (old_stem, old_display):
                if v and v not in aliases and v != new_stem:
                    aliases.append(v)
            si["aliases"] = aliases
            content = _shared.write_with_semantic_index(
                strip_legacy_semantic_index(content), si
            )
            # H1 display update — handles both the templated
            # "# Person: Old Name" and bare "# Old Name" forms. Only the FIRST
            # heading, only when it still carries the old display name.
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if line.startswith("# "):
                    if old_display.lower() in line.lower():
                        import re as _re
                        lines[i] = _re.sub(
                            _re.escape(old_display), new_name, line, flags=_re.IGNORECASE
                        )
                    break
            content = "\n".join(lines)
            if context:
                content = _append_user_context(content, context, "rename")
            abs_path.write_text(content, encoding="utf-8")

            repo.git.mv(str(rel), str(new_rel))
            commit = self._commit(repo, f"manage(rename): {old_stem} → {new_stem}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "rename",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "old_path": str(rel).replace("\\", "/"),
                "new_path": str(new_rel).replace("\\", "/"),
                "aliases_added": [old_stem, old_display],
                "summary": f"Renamed {old_stem} → {new_stem}.",
            }

    # ── EDIT (sync; raw markdown, expert mode) ────────────────────────────────

    def manage_edit(self, path: str, markdown: str) -> Dict[str, Any]:
        """Overwrite an entity file with raw markdown. Validation: non-empty,
        and the result must still carry a parseable semantic index (frontmatter
        or legacy block) so the entity stays scannable/indexed."""
        if not markdown or not markdown.strip():
            raise ManagementError("markdown must be non-empty")
        if _shared.extract_semantic_index(markdown) is None:
            raise ManagementError(
                "edited markdown has no parseable semantic index (frontmatter "
                "or legacy SEMANTIC INDEX block) — the entity would vanish "
                "from the index. Keep the frontmatter block."
            )
        with self._lock():
            repo = self._repo()
            abs_path, _ = _require_entity(self.repo_path, path, self.ontology)
            rel = str(abs_path.relative_to(self.repo_path)).replace("\\", "/")
            abs_path.write_text(markdown, encoding="utf-8")
            repo.git.add(rel)
            commit = self._commit(repo, f"manage(edit): {abs_path.stem}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "edit",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "path": rel,
                "summary": f"Edited {rel}.",
            }

    # ── ALIAS (sync) ──────────────────────────────────────────────────────────

    def manage_alias(self, path: str, aliases: List[str]) -> Dict[str, Any]:
        """Add aliases to an entity's semantic index (dedupe prevention)."""
        cleaned = [a.strip() for a in aliases if a and a.strip()]
        if not cleaned:
            raise ManagementError("aliases must be a non-empty list of strings")
        with self._lock():
            repo = self._repo()
            abs_path, _ = _require_entity(self.repo_path, path, self.ontology)
            rel = str(abs_path.relative_to(self.repo_path)).replace("\\", "/")
            content = abs_path.read_text(encoding="utf-8")
            si = _shared.extract_semantic_index(content) or {}
            existing = list(si.get("aliases") or [])
            added = [a for a in cleaned if a not in existing]
            if not added:
                return {
                    "status": "ok", "tool": "alias", "commits": [], "added": [],
                    "summary": "All aliases already present.",
                }
            si["aliases"] = existing + added
            abs_path.write_text(
                _shared.write_with_semantic_index(strip_legacy_semantic_index(content), si),
                encoding="utf-8",
            )
            repo.git.add(rel)
            commit = self._commit(repo, f"manage(alias): {abs_path.stem} + {len(added)}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "alias",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "added": added,
                "summary": f"Added {len(added)} alias(es) to {abs_path.stem}.",
            }

    # ── DELETE (sync) ─────────────────────────────────────────────────────────

    def manage_delete(self, path: str) -> Dict[str, Any]:
        """git rm an entity file (destructive; recoverable via git history)."""
        with self._lock():
            repo = self._repo()
            abs_path, _ = _require_entity(self.repo_path, path, self.ontology)
            rel = str(abs_path.relative_to(self.repo_path)).replace("\\", "/")
            repo.git.rm(rel)
            commit = self._commit(repo, f"manage(delete): {abs_path.stem}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "delete",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "deleted": rel,
                "summary": f"Deleted {rel} (recoverable from git history).",
            }

    # ── LINK (sync; deterministic bidirectional) ──────────────────────────────

    def manage_link(
        self, path: str, target_path: str, note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Connect two entities: SI related_entities (both directions) + a
        wikilink line under `## Related Links` (created on first use)."""
        with self._lock():
            repo = self._repo()
            a_abs, _ = _require_entity(self.repo_path, path, self.ontology)
            b_abs, _ = _require_entity(self.repo_path, target_path, self.ontology)
            if a_abs == b_abs:
                raise ManagementError("cannot link an entity to itself")
            commits = []
            for src, dst in ((a_abs, b_abs), (b_abs, a_abs)):
                content = src.read_text(encoding="utf-8")
                si = _shared.extract_semantic_index(content) or {}
                related = list(si.get("related_entities") or [])
                dst_stem = dst.stem
                if dst_stem not in related:
                    related.append(dst_stem)
                si["related_entities"] = related
                body = strip_legacy_semantic_index(content)

                link_line = f"- [[{dst.stem}]]"
                if note:
                    link_line += f" — {note.strip()}"
                if "## Related Links" in body:
                    if f"[[{dst.stem}]]" not in body:
                        head, sep, tail = body.partition("## Related Links")
                        body = head + sep + tail.rstrip("\n") + "\n" + link_line + "\n"
                else:
                    body = body.rstrip("\n") + f"\n\n## Related Links\n{link_line}\n"
                src.write_text(
                    _shared.write_with_semantic_index(body, si), encoding="utf-8"
                )
                repo.git.add(str(src.relative_to(self.repo_path)))
            commit = self._commit(repo, f"manage(link): {a_abs.stem} ↔ {b_abs.stem}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "link",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "linked": [str(a_abs.relative_to(self.repo_path)), str(b_abs.relative_to(self.repo_path))],
                "summary": f"Linked {a_abs.stem} ↔ {b_abs.stem}.",
            }

    # ── ADD-NOTE (LLM; weaves natural-language context into the body) ────────

    def manage_add_note(self, path: str, text: str) -> Dict[str, Any]:
        """Weave a user's natural-language note into the entity body via LLM.
        Git-only provenance (no timeline entry). The writer reads the full body
        on every future update, so the note steers later reprocessing."""
        if not text or not text.strip():
            raise ManagementError("text must be non-empty")
        with self._lock():
            repo = self._repo()
            abs_path, _ = _require_entity(self.repo_path, path, self.ontology)
            rel = str(abs_path.relative_to(self.repo_path)).replace("\\", "/")
            entity = self._load_entity(abs_path)
            tmpl = (self.prompts_path / "manage_note.txt").read_text(encoding="utf-8")
            prompt = tmpl.format(
                file_path=rel,
                file_content=entity["content"],
                user_note=text.strip(),
                today=datetime.now().strftime("%Y-%m-%d"),
            )
            merged = self._call_llm(prompt, False)
            if not isinstance(merged, str) or not merged.strip():
                raise RuntimeError("LLM returned empty note merge — entity unchanged")
            # Ingress normalization: never persist nested cue lists.
            si = _shared.extract_semantic_index(merged) or entity["semantic_index"]
            abs_path.write_text(
                _shared.write_with_semantic_index(
                    strip_legacy_semantic_index(merged),
                    normalize_semantic_index(dict(si)),
                ),
                encoding="utf-8",
            )
            repo.git.add(rel)
            commit = self._commit(repo, f"manage(note): {abs_path.stem}")
            index_commit = self._rebuild_index(repo)
            return {
                "status": "ok",
                "tool": "note",
                "commits": [commit] + ([index_commit] if index_commit else []),
                "path": rel,
                "summary": f"Wove user note into {abs_path.stem}.",
            }

    # ── SUGGESTIONS (sync; no LLM) ───────────────────────────────────────────

    def merge_suggestions(self, name_threshold: Optional[float] = None) -> Dict[str, Any]:
        """Dedupe review queue: candidate pairs from the (relaxed) prefilter —
        same-type, ANY-one-signal corroboration. No judge; the UI decides."""
        entities = _shared.scan_entities(
            self.repo_path, entity_dirs=self._entity_dirs_paths()
        )
        pairs = _dedupe.find_candidate_pairs(
            entities, name_threshold=name_threshold
        )
        suggestions = []
        for a, b in pairs:
            si_a, si_b = a["semantic_index"], b["semantic_index"]
            from difflib import SequenceMatcher
            name_a = str(si_a.get("name") or a["file"].stem)
            name_b = str(si_b.get("name") or b["file"].stem)
            suggestions.append({
                "a_path": a["path"],
                "b_path": b["path"],
                "a_name": name_a,
                "b_name": name_b,
                "a_type": si_a.get("type"),
                "b_type": si_b.get("type"),
                "name_similarity": round(
                    SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio(), 3
                ),
                "shared_related": sorted(
                    set(map(str.lower, si_a.get("related_entities") or []))
                    & set(map(str.lower, si_b.get("related_entities") or []))
                ),
                "shared_cues": sorted(
                    set(map(str.lower, si_a.get("hard_cues") or []))
                    & set(map(str.lower, si_b.get("hard_cues") or []))
                ),
            })
        suggestions.sort(key=lambda s: s["name_similarity"], reverse=True)
        return {
            "status": "ok",
            "tool": "merge_suggestions",
            "count": len(suggestions),
            "suggestions": suggestions,
            "summary": f"{len(suggestions)} candidate pair(s).",
        }
