#!/usr/bin/env python3
"""Split SWE-chat sessions into a development set (20%, for prompt iteration)
and a test set (80%, held out for the actual method run), stratified on
observable features, grouped by repository so no repo crosses the split.

Stratification priority (highest first): success-score bin, agent/model
configuration, repo type (dominant language by touched files), trajectory
length (turn count), tool-use intensity (tool call count). All five features
are aggregated to the REPO level (not session level) because the grouping
unit is the repo - a repo can't be half in dev and half in test, so the
thing being stratified is each repo's overall profile.

Assignment uses Taves' minimization method (the standard technique for
balancing multiple prognostic factors across two arms without a combinatorial
cross-product of factors - originally from clinical trial randomization,
adapted here with sessions as the balancing unit since repos vary in session
count from 1 to 870): repos are processed largest-first (the least flexible,
highest-impact items get placed while the most balancing room remains), and
each is assigned to whichever side currently has the largest weighted
deficit against the 20% target, summed across all five factors with
priority-ordered weights (each factor counts half as much as the one above
it). This directly targets each factor's MARGINAL dev-fraction, which a
single composite stratification key does not: an earlier version of this
script built one composite key (success x agent x language x length x
intensity, collapsing rare combinations into coarser prefixes) and split via
sklearn's StratifiedGroupKFold, which balanced the joint label but let
lower-priority-looking marginals swing wildly (e.g. one run put 0% of Codex
sessions and 85% of Gemini CLI sessions in dev) - excessive cross-
stratification by another name, since most repos ended up in a stratum of
size 1-2 and group-count imbalance dominated. Minimization avoids ever
forming that cross product.

Caveat this script prints explicitly: repo size in this corpus is extremely
skewed (one repo alone is ~15% of all sessions), so which side that single
repo lands on can pull the realized dev fraction and factor balance away
from the requested target even with a correct algorithm - inspect the
printed summary rather than assuming the request was hit exactly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/swe_chat_sessions.parquet")
DEFAULT_OUTPUT_DIR = Path("data/processed/splits")

KEY_COLUMNS_PRIORITY = ["success_bin", "agent_bucket", "repo_language", "length_bin", "intensity_bin"]

LANGUAGE_BY_EXTENSION = {
    "py": "python", "pyi": "python",
    "js": "javascript", "jsx": "javascript", "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "tsx": "typescript",
    "go": "go",
    "rs": "rust",
    "java": "java",
    "kt": "kotlin", "kts": "kotlin",
    "rb": "ruby",
    "php": "php",
    "c": "c", "h": "c",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
    "cs": "csharp",
    "swift": "swift",
    "sh": "shell", "bash": "shell",
    "scala": "scala",
    "ex": "elixir", "exs": "elixir",
}
NON_LANGUAGE_EXTENSIONS = {
    "md", "txt", "json", "yaml", "yml", "toml", "cfg", "ini", "lock", "svg",
    "png", "jpg", "jpeg", "gif", "csv", "html", "css", "scss", "xml", "gitignore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimization-balanced group split of SWE-chat sessions into dev/test.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-min-count", type=int, default=20, help="Below this session count, agent -> 'other'.")
    parser.add_argument("--language-min-repos", type=int, default=4, help="Below this repo count, language -> 'other'.")
    parser.add_argument(
        "--overall-fraction-weight",
        type=float,
        default=32.0,
        help="Weight on the global dev/test session ratio term, relative to per-factor weights (halving per priority rank below it).",
    )
    parser.add_argument(
        "--keep-unknown-success-or-agent",
        action="store_true",
        help="By default, sessions with unknown success score or unknown agent are dropped before splitting (success score is the primary stratification factor, so a session without one can't be meaningfully placed). Pass this to keep them instead.",
    )
    parser.add_argument(
        "--success-cutoff",
        type=float,
        default=70.0,
        help="Failure population cutoff: only sessions with success score strictly below this are kept (matches this repo's existing 'leq_70' convention and the dataset rubric's 70-89 = 'core goal achieved' band starting at this cutoff). Pass a value >=100 to disable.",
    )
    parser.add_argument(
        "--judge-ready",
        type=Path,
        default=Path("data/processed/swe_chat_sessions_judge_ready_full.parquet"),
        help="Parquet with a session_id column covering every session that actually has judge-ready content (e.g. output of prepare_judge_sessions.py). Sessions absent from it are dropped before splitting, so the split never contains a session with nothing to extract from. Pass an empty string to skip this filter.",
    )
    return parser.parse_args()


def success_bin(value: float) -> str:
    """Per the dataset's success rubric: <50 = primary goal not achieved,
    50-69 = significant subtask incomplete or known defect, 70-89 = core
    goal achieved with only minor issues. Boundaries are inclusive on the
    lower edge of each band (a score of exactly 50 is "50-69", not "<50")."""
    if pd.isna(value):
        return "unknown"
    if value < 50:
        return "lt50_primary_goal_not_achieved"
    if value < 70:
        return "50to69_significant_defect"
    return "geq70_core_goal_achieved"


def normalize_agent(value: Any) -> str:
    text = str(value or "unknown").strip()
    canonical = {"opencode": "OpenCode", "claude-code": "Claude Code"}
    return canonical.get(text.lower(), text) if text.lower() in canonical else text


def bucket_agent(agent_counts: Counter[str], min_count: int) -> dict[str, str]:
    return {agent: (agent if count >= min_count else "other") for agent, count in agent_counts.items()}


def dominant_language(files_touched_json: str) -> str | None:
    try:
        files = json.loads(files_touched_json) if isinstance(files_touched_json, str) else files_touched_json
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(files, list):
        return None
    ext_counts: Counter[str] = Counter()
    for path in files:
        if not isinstance(path, str) or "." not in path:
            continue
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in NON_LANGUAGE_EXTENSIONS:
            continue
        language = LANGUAGE_BY_EXTENSION.get(ext)
        if language:
            ext_counts[language] += 1
    if not ext_counts:
        return None
    return ext_counts.most_common(1)[0][0]


def quantile_bin(values: pd.Series, labels: list[str]) -> pd.Series:
    ranks = values.rank(method="first")
    try:
        return pd.qcut(ranks, q=len(labels), labels=labels)
    except ValueError:
        return pd.Series([labels[len(labels) // 2]] * len(values), index=values.index)


def build_repo_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df["success_bin"] = pd.to_numeric(df["session_success"], errors="coerce").map(success_bin)
    df["agent_norm"] = df["agent"].map(normalize_agent)
    agent_map = bucket_agent(Counter(df["agent_norm"]), args.agent_min_count)
    df["agent_bucket"] = df["agent_norm"].map(agent_map)
    df["session_language"] = df["files_touched"].map(dominant_language)

    repo_language_votes = (
        df.dropna(subset=["session_language"])
        .groupby("repo_id")["session_language"]
        .agg(lambda s: Counter(s).most_common(1)[0][0])
    )
    language_repo_counts = repo_language_votes.value_counts()
    rare_languages = set(language_repo_counts[language_repo_counts < args.language_min_repos].index)
    repo_language = repo_language_votes.map(lambda lang: "other" if lang in rare_languages else lang)
    repo_language = repo_language.reindex(df["repo_id"].unique()).fillna("unknown")

    repo_df = pd.DataFrame(
        {
            "success_bin": df.groupby("repo_id")["success_bin"].agg(lambda s: Counter(s).most_common(1)[0][0]),
            "agent_bucket": df.groupby("repo_id")["agent_bucket"].agg(lambda s: Counter(s).most_common(1)[0][0]),
            "repo_language": repo_language,
            "turn_median": df.groupby("repo_id")["turn_count"].median(),
            "tool_median": df.groupby("repo_id")["tool_call_count"].median(),
            "session_count": df.groupby("repo_id").size(),
        }
    )
    repo_df["length_bin"] = quantile_bin(repo_df["turn_median"], ["short", "medium", "long"])
    repo_df["intensity_bin"] = quantile_bin(repo_df["tool_median"], ["low", "medium", "high"])
    return repo_df.reset_index().rename(columns={"index": "repo_id"})


def minimization_assign(
    repo_df: pd.DataFrame,
    dev_fraction: float,
    seed: int,
    overall_weight: float,
) -> pd.Series:
    """Taves' minimization method, weighted by session count. Higher-priority
    factors get exponentially larger weights (halving per rank) so the
    algorithm always prefers fixing a high-priority imbalance over a
    low-priority one when the two conflict."""
    weights = {col: overall_weight / (2 ** (i + 1)) for i, col in enumerate(KEY_COLUMNS_PRIORITY)}
    total_sessions = repo_df["session_count"].sum()
    target_dev_by_factor = {
        col: (repo_df.groupby(col)["session_count"].sum() * dev_fraction) for col in KEY_COLUMNS_PRIORITY
    }
    dev_count_by_factor = {col: pd.Series(0.0, index=target_dev_by_factor[col].index) for col in KEY_COLUMNS_PRIORITY}
    dev_total = 0.0
    target_dev_total = total_sessions * dev_fraction

    # Seeded-random order, not largest-first: with repo size this skewed (1 to
    # 870 sessions), placing the biggest repos first lets 2-3 of them alone
    # saturate the dev session target, after which every remaining repo scores
    # as "test-preferred" and dev ends up with almost no repo diversity - bad
    # for a set meant to be iterated on by hand. Random order lets the target
    # build up gradually across many repos; minimization still corrects
    # imbalances as they arise regardless of arrival order.
    rng = np.random.default_rng(seed)
    order = repo_df.sample(frac=1.0, random_state=seed).index
    assignment = pd.Series(index=repo_df.index, dtype=object)
    for idx in order:
        row = repo_df.loc[idx]
        w = row["session_count"]

        def cost(assign_to_dev: bool) -> float:
            delta_total = w if assign_to_dev else 0.0
            c = overall_weight * abs((dev_total + delta_total) - target_dev_total)
            for col in KEY_COLUMNS_PRIORITY:
                level = row[col]
                delta = w if assign_to_dev else 0.0
                current = dev_count_by_factor[col].get(level, 0.0)
                c += weights[col] * abs((current + delta) - target_dev_by_factor[col].get(level, 0.0))
            return c

        cost_dev, cost_test = cost(True), cost(False)
        if abs(cost_dev - cost_test) < 1e-9:
            choose_dev = bool(rng.integers(0, 2))
        else:
            choose_dev = cost_dev < cost_test

        assignment.loc[idx] = "dev" if choose_dev else "test"
        if choose_dev:
            dev_total += w
            for col in KEY_COLUMNS_PRIORITY:
                level = row[col]
                dev_count_by_factor[col][level] = dev_count_by_factor[col].get(level, 0.0) + w

    return assignment


def realized_fraction_table(sessions: pd.DataFrame, dev_mask: pd.Series, columns: list[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for col in columns:
        dev_counts = sessions.loc[dev_mask, col].value_counts()
        all_counts = sessions[col].value_counts()
        table[col] = {
            str(level): {
                "dev": int(dev_counts.get(level, 0)),
                "total": int(all_counts.get(level, 0)),
                "dev_fraction": round(dev_counts.get(level, 0) / all_counts[level], 4) if all_counts.get(level, 0) else None,
            }
            for level in all_counts.index
        }
    return table


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input)
    df = df[df["repo_id"].notna()].copy()

    df = df[~df["session_id"].str.startswith("ses_test")].copy()

    if args.judge_ready is not None and str(args.judge_ready) not in ("", "."):
        judge_ready_ids = set(pd.read_parquet(args.judge_ready, columns=["session_id"])["session_id"])
        not_judgeable = ~df["session_id"].isin(judge_ready_ids)
        dropped_not_judgeable = int(not_judgeable.sum())
        if dropped_not_judgeable:
            print(
                f"Dropping {dropped_not_judgeable} sessions with no judge-ready content "
                f"(no assistant response, or otherwise absent from {args.judge_ready}) out of {len(df)}."
            )
            df = df[~not_judgeable].copy()
    else:
        dropped_not_judgeable = 0

    dropped_unknown = 0
    if not args.keep_unknown_success_or_agent:
        unknown_success = pd.to_numeric(df["session_success"], errors="coerce").isna()
        unknown_agent = df["agent"].map(normalize_agent).str.lower() == "unknown"
        drop_mask = unknown_success | unknown_agent
        dropped_unknown = int(drop_mask.sum())
        print(
            f"Dropping {dropped_unknown} sessions with unknown success score or unknown agent "
            f"({int(unknown_success.sum())} unknown success, {int(unknown_agent.sum())} unknown agent, "
            f"{int((unknown_success & unknown_agent).sum())} both) out of {len(df)}."
        )
        df = df[~drop_mask].copy()

    dropped_not_failure = 0
    if args.success_cutoff < 100:
        not_failure = pd.to_numeric(df["session_success"], errors="coerce") >= args.success_cutoff
        dropped_not_failure = int(not_failure.sum())
        print(f"Dropping {dropped_not_failure} sessions with success score >= {args.success_cutoff} (not failures) out of {len(df)}.")
        df = df[~not_failure].copy()

    repo_df = build_repo_features(df, args)
    repo_df["split"] = minimization_assign(repo_df, args.dev_fraction, args.seed, args.overall_fraction_weight)

    # build_repo_features already added session-level success_bin/agent_bucket/session_language
    # to df in place; only merge back the columns that exist solely at repo granularity
    # (repo_language/length_bin/intensity_bin/split), so the audit below reports genuine
    # per-session ground truth for success/agent rather than the repo-level mode used to
    # make the (necessarily repo-level) split decision.
    df = df.merge(
        repo_df[["repo_id", "split", "repo_language", "length_bin", "intensity_bin"]], on="repo_id", how="left"
    )
    dev_mask = df["split"] == "dev"

    dev_repos = set(df.loc[dev_mask, "repo_id"])
    test_repos = set(df.loc[~dev_mask, "repo_id"])
    assert not (dev_repos & test_repos), f"repo(s) leaked across split: {dev_repos & test_repos}"

    drop_cols = [
        "split", "success_bin", "agent_norm", "agent_bucket", "session_language",
        "repo_language", "length_bin", "intensity_bin",
    ]
    dev_df = df[dev_mask].drop(columns=drop_cols)
    test_df = df[~dev_mask].drop(columns=drop_cols)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dev_path = args.output_dir / "swe_chat_dev.parquet"
    test_path = args.output_dir / "swe_chat_test.parquet"
    dev_df.to_parquet(dev_path, index=False)
    test_df.to_parquet(test_path, index=False)

    largest_repos = repo_df.sort_values("session_count", ascending=False).head(5)
    largest_repos_report = [
        {
            "repo_id": row.repo_id,
            "session_count": int(row.session_count),
            "share_of_corpus": round(row.session_count / len(df), 4),
            "split": row.split,
        }
        for row in largest_repos.itertuples()
    ]

    summary = {
        "input": str(args.input),
        "method": "minimization (Taves), session-count-weighted, priority-ordered factor weights",
        "dropped_not_judge_ready_sessions": dropped_not_judgeable,
        "dropped_unknown_success_or_agent_sessions": dropped_unknown,
        "dropped_not_failure_sessions_geq_cutoff": dropped_not_failure,
        "success_cutoff": args.success_cutoff,
        "total_sessions": len(df),
        "total_repos": df["repo_id"].nunique(),
        "requested_dev_fraction": args.dev_fraction,
        "realized_dev_sessions": int(dev_mask.sum()),
        "realized_test_sessions": int((~dev_mask).sum()),
        "realized_dev_fraction": round(dev_mask.mean(), 4),
        "realized_dev_repos": len(dev_repos),
        "realized_test_repos": len(test_repos),
        "stratification_priority": KEY_COLUMNS_PRIORITY,
        "largest_repos_and_which_split_they_landed_in": largest_repos_report,
        "caveat": (
            "Repo size is extremely skewed (see largest_repos_... above); which split the "
            "largest repos land in can pull realized fractions away from the request even "
            "with correct grouping/balancing. This is expected, not a bug."
        ),
        "balance_by_feature": realized_fraction_table(df, dev_mask, KEY_COLUMNS_PRIORITY),
        "dev_path": str(dev_path),
        "test_path": str(test_path),
    }
    summary_path = args.output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "balance_by_feature"}, indent=2, default=str))


if __name__ == "__main__":
    main()
