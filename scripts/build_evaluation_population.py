#!/usr/bin/env python3
"""Build the finite reference population every real-data statistical evaluation
samples from.

The population is every judged session that is not in dev:

- successes: sessions with session_success >= 70 (the dataset rubric's success
  cutoff) that survive the same corpus filters as the failure population
  (non-null repo, not a `ses_test*` placeholder, judge-ready content, known
  success score and agent) -- see scripts/build_dev_test_split.py;
- failed test sessions: the 993 sessions the failure-mode judge ran on
  (the 995-session test split minus the two dropped for exceeding the judge's
  context window) -- taken from the judge output, not the split, so a session
  the judge never saw cannot enter the population.

Test sessions the judge found no failure in, and all successes, are absent from
every taxonomy; the evaluation runners assign them the success sentinel (never
novel, never "missing" mass). Dev sessions are excluded entirely.

    uv run python scripts/build_evaluation_population.py \\
        --output data/population/test_population.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dev_test_split import normalize_agent  # noqa: E402

SUCCESS_CUTOFF = 70

SESSIONS_PARQUET = Path("data/processed/swe_chat_sessions.parquet")
JUDGE_READY_PARQUET = Path("data/processed/swe_chat_sessions_judge_ready_full.parquet")
DEV_PARQUET = Path("data/processed/splits/swe_chat_dev.parquet")
TEST_PARQUET = Path("data/processed/splits/swe_chat_test.parquet")
JUDGED_TEST_JSONLS = (
    Path("data/failure_modes/primary_test993.jsonl"),
    Path("data/failure_modes/set_valued_test993.jsonl"),
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required input does not exist: {path}")


def load_judged_session_ids(path: Path) -> set[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["session_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate session_ids")
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (SESSIONS_PARQUET, JUDGE_READY_PARQUET, DEV_PARQUET, TEST_PARQUET, *JUDGED_TEST_JSONLS):
        require_file(path)
    if args.output.exists():
        raise FileExistsError(f"{args.output} already exists; refusing to overwrite")

    df = pd.read_parquet(SESSIONS_PARQUET)
    df = df[df["repo_id"].notna()]
    df = df[~df["session_id"].str.startswith("ses_test")]
    judge_ready_ids = set(pd.read_parquet(JUDGE_READY_PARQUET, columns=["session_id"])["session_id"])
    df = df[df["session_id"].isin(judge_ready_ids)]
    score = pd.to_numeric(df["session_success"], errors="coerce")
    unknown = score.isna() | (df["agent"].map(normalize_agent).str.lower() == "unknown")
    df, score = df[~unknown], score[~unknown]
    if df["session_id"].duplicated().any():
        raise ValueError("duplicate session_id after corpus filters")

    success_ids = set(df.loc[score >= SUCCESS_CUTOFF, "session_id"])
    failure_ids = set(df.loc[score < SUCCESS_CUTOFF, "session_id"])

    dev_ids = set(pd.read_parquet(DEV_PARQUET, columns=["session_id"])["session_id"])
    test_ids = set(pd.read_parquet(TEST_PARQUET, columns=["session_id"])["session_id"])
    if failure_ids != dev_ids | test_ids:
        raise ValueError("recomputed failure population does not equal dev + test split")
    if dev_ids & test_ids:
        raise ValueError("dev and test splits overlap")

    judged = [load_judged_session_ids(path) for path in JUDGED_TEST_JSONLS]
    if any(ids != judged[0] for ids in judged[1:]):
        raise ValueError(f"judged test sessions differ across {[str(p) for p in JUDGED_TEST_JSONLS]}")
    judged_test_ids = judged[0]
    if not judged_test_ids <= test_ids:
        raise ValueError("judged sessions include sessions outside the test split")
    unjudged_test_ids = sorted(test_ids - judged_test_ids)

    population = sorted(success_ids | judged_test_ids)
    if success_ids & judged_test_ids:
        raise ValueError("a session is both a success and a judged failed test session")
    if set(population) & dev_ids:
        raise ValueError("population contains dev sessions")

    record = {
        "description": (
            "Finite reference population for the statistical evaluation: all non-dev sessions "
            f"with session_success >= {SUCCESS_CUTOFF} plus every failed test session the "
            "failure-mode judge ran on. Built by scripts/build_evaluation_population.py."
        ),
        "success_cutoff": SUCCESS_CUTOFF,
        "n_sessions": len(population),
        "n_success_sessions": len(success_ids),
        "n_failed_test_sessions": len(judged_test_ids),
        "n_dev_sessions_excluded": len(dev_ids),
        "unjudged_test_sessions_excluded": unjudged_test_ids,
        "session_ids": population,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {args.output}: {len(population)} sessions "
          f"({len(success_ids)} successes + {len(judged_test_ids)} failed test sessions; "
          f"{len(dev_ids)} dev and {len(unjudged_test_ids)} unjudged test sessions excluded)")


if __name__ == "__main__":
    main()
