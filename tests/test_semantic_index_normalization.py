# CAPABILITY: Regression tests for semantic-index list normalization.
# INPUTS: Nested-list LLM output shapes seen in ChatBarry production
#         (e.g. hard_cues: ["a", "b", ["c", "d"]]).
# OUTPUTS: Proof that (1) parsing normalizes poisoned files, (2) writes never
#          persist nesting, (3) the redistribute/link report builders and the
#          dedupe prefilter survive the shapes that used to crash them with
#          TypeError, (4) scan_entities + master-index rebuild emit flat data.
# CONSTRAINTS: No network, no real LLM. Mirrors the _fixtures conventions.

from __future__ import annotations

import json
from pathlib import Path

from tests._fixtures import build_worktree

from diffmem.consolidator_agent import _dedupe, _link, _redistribute, _shared
from diffmem.frontmatter import (
    SI_LIST_FIELDS,
    flatten_str_list,
    normalize_semantic_index,
    parse_frontmatter,
)

# The exact shape that crashed ~80% of consolidate runs in ChatBarry prod:
NESTED = ["alpha", "beta", ["gamma", "delta"]]
DEEPLY_NESTED = ["alpha", [["beta", ["gamma"]], "delta"], None, "", 7]


# --- flatten_str_list unit behavior -------------------------------------------


def test_flatten_nested_list():
    assert flatten_str_list(NESTED) == ["alpha", "beta", "gamma", "delta"]


def test_flatten_deeply_nested_dedupes_and_stringifies():
    # None + "" dropped, int stringified, order preserved.
    assert flatten_str_list(DEEPLY_NESTED) == ["alpha", "beta", "gamma", "delta", "7"]


def test_flatten_dedupes_preserving_first_seen():
    assert flatten_str_list(["a", ["a", "b"], "b"]) == ["a", "b"]


def test_flatten_scalar_and_empty():
    assert flatten_str_list("solo") == ["solo"]
    assert flatten_str_list([]) == []
    assert flatten_str_list(None) == []


def test_flatten_strips_whitespace():
    assert flatten_str_list([" a ", ["b  "]]) == ["a", "b"]


# --- normalize_semantic_index ---------------------------------------------------


def test_normalize_all_list_fields():
    si = {f: NESTED for f in SI_LIST_FIELDS}
    si["type"] = "human"
    si["name"] = "Maya"
    out = normalize_semantic_index(si)
    for f in SI_LIST_FIELDS:
        assert out[f] == ["alpha", "beta", "gamma", "delta"], f
    # Unknown / scalar fields pass through untouched.
    assert out["type"] == "human"
    assert out["name"] == "Maya"


def test_normalize_missing_fields_untouched():
    si = {"name": "Maya", "type": "human"}
    assert normalize_semantic_index(si) == {"name": "Maya", "type": "human"}


def test_normalize_non_dict():
    assert normalize_semantic_index(None) == {}
    assert normalize_semantic_index("junk") == {}


def test_normalize_none_valued_field():
    si = {"hard_cues": None, "name": "Maya"}
    out = normalize_semantic_index(si)
    assert out["hard_cues"] == []


# --- read choke point: extract_semantic_index -----------------------------------


def _entity_file(nested_cues) -> str:
    """Frontmatter (v2) entity file whose hard_cues came from a careless LLM."""
    return (
        "---\n"
        "name: Maya\n"
        "type: human\n"
        f"hard_cues: {json.dumps(nested_cues)}\n"
        "related_entities: [\"alex\"]\n"
        "---\n"
        "# Maya\n\n## Profile\n- VP of Technology.\n"
    )


def test_extract_normalizes_frontmatter_nested_lists():
    si = _shared.extract_semantic_index(_entity_file(NESTED))
    assert si["hard_cues"] == ["alpha", "beta", "gamma", "delta"]


def test_extract_normalizes_legacy_json_block_nested_lists():
    legacy = (
        "# Maya\n\n## Profile\n- VP.\n\n## SEMANTIC INDEX\n"
        + json.dumps({"name": "Maya", "type": "human", "hard_cues": NESTED})
        + "\n"
    )
    si = _shared.extract_semantic_index(legacy)
    assert si["hard_cues"] == ["alpha", "beta", "gamma", "delta"]


# --- write choke point: write_with_semantic_index --------------------------------


def test_write_never_persists_nested_lists(tmp_path: Path):
    body = "# Maya\n\n## Profile\n- VP.\n"
    si = {"name": "Maya", "type": "human", "hard_cues": NESTED, "aliases": [["M", ["Maya B"]]]}
    written = _shared.write_with_semantic_index(body, si)
    fm, _ = parse_frontmatter(written)
    assert fm["hard_cues"] == ["alpha", "beta", "gamma", "delta"]
    assert fm["aliases"] == ["M", "Maya B"]
    # Round-trip through the read choke point is stable.
    assert _shared.extract_semantic_index(written)["hard_cues"] == fm["hard_cues"]


# --- end-to-end: the exact production crash sites --------------------------------


def test_redistribute_candidates_block_survives_nested_cues(tmp_path: Path):
    """_redistribute._candidates_block joined nested hard_cues -> TypeError (prod crash #1)."""
    wt = build_worktree(tmp_path)
    ent = {
        "file": wt / "memories" / "people" / "maya.md",
        "path": "memories/people/maya.md",
        "tokens": 10,
        "semantic_index": _shared.extract_semantic_index(_entity_file(NESTED)),
    }
    block = _redistribute._candidates_block([ent], exclude=wt / "nonexistent.md")
    assert "alpha,beta,gamma,delta" in block


def test_link_cooccurrence_block_survives_nested_cues(tmp_path: Path):
    """_link._cooccurrence_block joined nested hard_cues -> TypeError (prod crash #2)."""
    wt = build_worktree(tmp_path)
    target = wt / "memories" / "people" / "maya.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_entity_file(NESTED), encoding="utf-8")
    block, targets = _link._cooccurrence_block(wt, {"memories/people/maya.md": 3})
    assert targets == ["memories/people/maya.md"]
    assert "alpha,beta,gamma,delta" in block


def test_dedupe_prefilter_survives_nested_cues(tmp_path: Path):
    """_dedupe._overlap mapped str.lower over nested lists -> TypeError (latent crash #3)."""
    wt = build_worktree(tmp_path)
    maya = wt / "memories" / "people" / "maya.md"
    maya2 = wt / "memories" / "people" / "maya_b.md"
    ents = [
        {
            "file": maya,
            "path": "memories/people/maya.md",
            "semantic_index": normalize_semantic_index(
                {"hard_cues": NESTED, "related_entities": NESTED, "name": "Maya", "type": "human"}
            ),
        },
        {
            "file": maya2,
            "path": "memories/people/maya_b.md",
            "semantic_index": normalize_semantic_index(
                {"hard_cues": ["gamma"], "related_entities": ["beta"], "name": "Maya B", "type": "human"}
            ),
        },
    ]
    # Must not raise; post-flatten the pair shares cues + related overlap, so
    # it is surfaced as a dedupe candidate (name similarity 1.0 >= 0.8).
    pairs = _dedupe.find_candidate_pairs(ents)
    assert (ents[0], ents[1]) in pairs or (ents[1], ents[0]) in pairs


def test_scan_entities_and_master_index_emit_flat_data(tmp_path: Path):
    """scan_entities + rebuild_master_index fed nested cues into index.md JSON
    (poisoning the writer's identify step). Both must emit flat lists now."""
    wt = build_worktree(tmp_path)
    people = wt / "memories" / "people"
    (people / "nested.md").write_text(_entity_file(NESTED), encoding="utf-8")

    entities = _shared.scan_entities(wt)
    si = next(e["semantic_index"] for e in entities if e["path"].endswith("nested.md"))
    assert si["hard_cues"] == ["alpha", "beta", "gamma", "delta"]

    import git

    repo = git.Repo(wt)
    repo.git.add("-A")
    repo.index.commit("fixture")
    index_path = _shared.rebuild_master_index(wt, "alex", repo=repo)
    raw = index_path.read_text(encoding="utf-8")

    # index.md embeds one JSON blob per entity on a ``` fenced line — find
    # the nested.md one and assert its cues are flat.
    blobs = [
        json.loads(line.strip("`"))
        for line in raw.splitlines()
        if line.startswith("```") and line.strip("`").startswith("{")
    ]
    nested_blob = next(b for b in blobs if b.get("file") == "memories/people/nested.md")
    assert nested_blob["hard_cues"] == ["alpha", "beta", "gamma", "delta"]


def test_llm_index_response_normalized_before_write(tmp_path: Path):
    """Redistribute's _build_semantic_index returns the LLM dict verbatim when
    it has a name; write_with_semantic_index must flatten it before disk (ingress)."""
    wt = build_worktree(tmp_path)
    bad_llm = {"name": "Orphan Theme", "type": "concept", "hard_cues": NESTED}
    out = _shared.write_with_semantic_index("# Theme\n\n- Some prose.\n", bad_llm)
    fm, _ = parse_frontmatter(out)
    assert fm["hard_cues"] == ["alpha", "beta", "gamma", "delta"]
