# `data/`

Layout, and which stage writes each directory. See
[experimental_writeup/data_preprocessing.md](../experimental_writeup/data_preprocessing.md) and
[experimental_writeup/failure_mode_pipeline.md](../experimental_writeup/failure_mode_pipeline.md)
for the pipeline these correspond to.

```
data/
  processed/        canonical datasets: the raw session join, the judge-ready parquet, dev/test splits
  api_responses/    exact request/response records for every paid API run, one dir per run
  failure_modes/    parsed per-session labels -- what downstream stages read
  taxonomy/         clustered fixed-taxonomy outputs built from data/failure_modes/
  plots/            generated figures
```

## `processed/`

Canonical inputs. `swe_chat_sessions.parquet` (raw join),
`swe_chat_sessions_judge_ready_full.parquet` (the `judge_input_json` field every judge reads), and
`splits/` (dev/test assignment plus `split_summary.json`). Parquet files here are tracked via Git
LFS.

## `api_responses/<run_label>/`

One directory per paid run, holding the full record so results stay auditable and reproducible:
`payloads_<model>.jsonl` (exact requests), `responses_<model>.jsonl` (exact responses, including
echoed prompt text), `response_ids_<model>.jsonl`, and `skipped.jsonl` where a run excluded
sessions. These files are large and are not meant to be read directly by analysis code — use
`data/failure_modes/` instead.

One directory per run, all belonging to the current one-shot pipeline. Superseded runs from the
two-stage Qwen pipeline, and dev runs of earlier drafts of the current prompt, are kept on the
`main` branch only.

## `failure_modes/`

Per-session labels parsed out of a run by `scripts/extract_failure_modes.py`, one JSONL row per
session plus a `.manifest.json` recording the source run, counts, and any skipped error rows.

Both representations use one envelope, so they load the same way:

| field | |
|---|---|
| `session_id`, `split` | session and dev/test assignment |
| `representation` | `primary` (one mode per session) or `set_valued` (all distinct modes) |
| `session_contains_failure` | true iff `failure_modes` is non-empty |
| `failure_modes` | list of modes, each with `canonical_name`, `definition`, `canonical_mitigation`, `supporting_evidence`, `confidence` (and `selection_reason` for primary) |
| `other_modes_considered` | primary runs only |

`supporting_evidence` holds verbatim excerpts from the session record, so a label can be checked
without reopening the raw responses file
(`scripts/validate_oneshot_supporting_evidence.py` does this).

## `taxonomy/`

Clustered outputs built from `data/failure_modes/` — the fixed taxonomy each representation's labels
collapse into. `primary_test_taxonomy.json` (538 modes) and `set_valued_test_taxonomy.json`
(1,472 modes), both at threshold 0.70.
