"""Shared embedding-text construction for the failure-mode clustering stage.
Used by scripts/embed_failure_modes.py, the only place that calls the paid
OpenAI embeddings API. Keeping the text construction in one place means a
consumer can detect a stale cache (built under a different text scheme)
instead of silently using embedding-mismatched vectors.

Schema note: the four taxonomies' Stage B prompts don't share one field set.
`variant1`/`v2_broad` use canonical_name/definition/canonical_mitigation;
`regression_scenario` uses canonical_name/trigger/expected_response/
failure_criterion/variable_inputs; `workflow_obligation` uses canonical_name/
trigger/required_action_or_evidence/allowed_next_step. Each mapping below
embeds canonical_name plus whichever fields define the failure's scope and
its correct/required behavior, and excludes pure parameter/metadata fields
(e.g. regression_scenario's variable_inputs) the same way slug was excluded
from the first schema.
"""
import gzip
import json
from pathlib import Path

_FIELDS_BY_SCHEMA: dict[str, tuple[str, ...]] = {
    "definition_mitigation": ("canonical_name", "definition", "canonical_mitigation"),
    "regression_scenario": ("canonical_name", "trigger", "expected_response", "failure_criterion"),
    "workflow_obligation": ("canonical_name", "trigger", "required_action_or_evidence", "allowed_next_step"),
}


def detect_schema(raw_assignment: dict) -> str:
    for schema_name, fields in _FIELDS_BY_SCHEMA.items():
        if all(field in raw_assignment for field in fields):
            return schema_name
    raise ValueError(
        f"raw assignment {raw_assignment.get('raw_id')!r} matches none of the known field "
        f"schemas {list(_FIELDS_BY_SCHEMA)} -- its keys are {sorted(raw_assignment)}; "
        f"add a new schema mapping to failure_mode_embedding_utils rather than guessing"
    )


def embedding_text(raw_assignment: dict) -> str:
    """The text embedded for one raw assignment: canonical_name plus the
    fields (specific to this taxonomy's Stage B schema) that define the
    failure's scope and its correct/required behavior -- see module
    docstring. For the definition_mitigation schema this is canonical_name +
    definition + canonical_mitigation. Including the mitigation distinguishes
    failures with similar names and definitions but different required behavior.

    Exact format is preserved per schema (not a generic join) so that
    changing this function for one schema doesn't retroactively mark every
    other schema's already-cached, already-paid-for embeddings as stale.
    """
    schema = detect_schema(raw_assignment)
    if schema == "definition_mitigation":
        return (
            f"{raw_assignment['canonical_name']}. {raw_assignment['definition']} "
            f"Mitigation: {raw_assignment['canonical_mitigation']}"
        )
    if schema == "regression_scenario":
        return (
            f"{raw_assignment['canonical_name']}. Trigger: {raw_assignment['trigger']} "
            f"Expected response: {raw_assignment['expected_response']} "
            f"Failure criterion: {raw_assignment['failure_criterion']}"
        )
    if schema == "workflow_obligation":
        return (
            f"{raw_assignment['canonical_name']}. Trigger: {raw_assignment['trigger']} "
            f"Required action or evidence: {raw_assignment['required_action_or_evidence']} "
            f"Allowed next step: {raw_assignment['allowed_next_step']}"
        )
    raise AssertionError(f"unhandled schema {schema!r} -- _FIELDS_BY_SCHEMA and embedding_text are out of sync")


def load_all_raw_assignments(raw_labels_json: Path) -> tuple[list[str], list[str]]:
    """Return (raw_ids, embedding_texts) for EVERY raw Stage B assignment in
    a taxonomy's raw_labels.json -- the full set-valued output (all A1, A2,
    ... per session), not just whichever one single-failure-mode selection
    later picks as the session's winner. This is what the embedding step
    (scripts/embed_failure_modes.py) embeds: caching the whole raw set
    means the cache stays valid even if the single-failure-mode selection
    rule changes which raw_id wins for a session, and it's reusable for
    future set-valued (not just single-mode) analyses.
    """
    raw_labels = json.loads(raw_labels_json.read_text())
    if not raw_labels:
        raise ValueError(f"{raw_labels_json} contains no raw assignments")
    raw_ids = [r["raw_id"] for r in raw_labels]
    texts = [embedding_text(r) for r in raw_labels]
    return raw_ids, texts


def resolve_primary_records_texts(records_json: Path) -> tuple[list[str], list[str]]:
    """Return (session_ids, embedding_texts) for a primary records file, in
    sorted session_id order -- the input shape the online-primary pipeline
    scores. The primary representation emits exactly one record per session
    and sets raw_id = session_id (scripts/build_failure_mode_records.py), so
    one id serves as both the session key and the embeddings-cache key.

    Raises on a session carrying more than one record: that is the set-valued
    shape, which this one-mode-per-draw pipeline cannot represent.
    """
    records = json.loads(records_json.read_text())
    if not records:
        raise ValueError(f"{records_json} contains no records")

    records.sort(key=lambda r: r["session_id"])
    session_ids = [r["session_id"] for r in records]

    duplicated = {s for s in session_ids if session_ids.count(s) > 1}
    if duplicated:
        raise ValueError(
            f"{records_json} has {len(duplicated)} session(s) with more than one record, "
            f"e.g. {sorted(duplicated)[0]!r} -- the online-primary pipeline requires exactly "
            f"one failure mode per session"
        )

    mismatched = [r["session_id"] for r in records if r["raw_id"] != r["session_id"]]
    if mismatched:
        raise ValueError(
            f"{records_json} has {len(mismatched)} record(s) whose raw_id differs from its "
            f"session_id, e.g. {mismatched[0]!r} -- expected the primary representation"
        )

    return session_ids, [embedding_text(r) for r in records]


def load_embeddings_cache(embeddings_cache_json: Path) -> dict[str, dict]:
    """Read an embeddings cache. A `.gz` suffix is read as gzip, anything else
    as plain JSON -- the committed caches under data/embeddings/ are gzipped
    because the raw JSON runs to tens of megabytes."""
    if not embeddings_cache_json.exists():
        return {}
    if embeddings_cache_json.suffix == ".gz":
        return json.loads(gzip.decompress(embeddings_cache_json.read_bytes()))
    return json.loads(embeddings_cache_json.read_text())
