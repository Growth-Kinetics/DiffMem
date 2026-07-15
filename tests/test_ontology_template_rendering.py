"""Regression: every ontology prompt template must .format() with ONLY the
kwargs the agent code actually passes.

Business rule protected: literal placeholders meant for the LLM (e.g. {Name},
{CompanyName}, {YYYY-MM-DD} in example frontmatter) MUST be brace-escaped
({{Name}}) in templates, or str.format() raises KeyError at runtime — which
crashed corporate-ontology onboarding/entity-creation in production
(diffMem-platform GOAL 2026-07-14-001; the deploy dodged it by switching to
the personal ontology, violating the platform's no-workarounds constraint).

If this test fails, fix the TEMPLATE (escape literals), never the caller.
"""
import re
from pathlib import Path

import pytest

ONTOLOGIES_DIR = Path(__file__).parent.parent / "src" / "diffmem" / "ontologies"

# Source of truth: the kwargs each agent call site passes to template.format().
# Keep in sync with writer_agent/{agent,onboarding_agent}.py + retrieval_agent/agent.py.
PASSED_KWARGS = {
    "onboard_user_entity": {"user_id", "user_info"},
    "onboard_identify_entities": {"user_id", "user_info"},
    "onboard_timeline_entry": {"session_date", "session_id", "user_id"},
    "1_identify_entities": {"memory_input", "semantic_index"},
    "2_create_entity_file": {"entity_name", "entity_summary", "example_content",
                             "example_file_name", "memory_input"},
    "3_update_entity_file": {"file_path_name", "file_content", "memory_input"},
    "4_create_timeline_entry": {"diff_text", "memory_input", "session_date", "session_id"},
    "build_index": {"file_content", "file_path", "last_update",
                    "memory_strength", "number_of_edits"},
    "build_followups": {"commitment_slugs", "memory_input", "previous_other_items"},
    "system": {"user_id", "baseline_tokens", "remaining_budget", "folder_listing"},
}

TEMPLATES = sorted(
    p for p in ONTOLOGIES_DIR.glob("*/prompts/*.txt") if p.stem in PASSED_KWARGS
)


@pytest.mark.parametrize("template_path", TEMPLATES, ids=lambda p: f"{p.parent.parent.name}/{p.name}")
def test_template_formats_with_only_passed_kwargs(template_path: Path):
    kwargs = {k: f"<{k}>" for k in PASSED_KWARGS[template_path.stem]}
    try:
        rendered = template_path.format_map  # noqa: B018 — attribute check only
    except AttributeError:
        pass
    rendered = template_path.read_text(encoding="utf-8").format(**kwargs)
    assert rendered  # non-empty render, no KeyError/IndexError


def test_no_unescaped_stray_tokens():
    """Belt-and-braces: no single-brace token outside the passed-kwargs set."""
    offenders = []
    for p in TEMPLATES:
        toks = set(re.findall(r"(?<!\{)\{([^{}\n]+)\}(?!\})", p.read_text(encoding="utf-8")))
        bad = toks - PASSED_KWARGS[p.stem]
        if bad:
            offenders.append(f"{p.parent.parent.name}/{p.name}: {sorted(bad)}")
    assert not offenders, "Escape these as {{token}}: " + "; ".join(offenders)
