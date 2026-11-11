"""
Online pairwise discovery statistics.

- Failure-mode representation: one failure mode per session
- Novelty rule: cosine similarity to previously discovered mode representatives
- Sampling: uniform with replacement from a finite population of sessions

At index t:
- z[t] = 1 iff draw t does not match any mode discovered before t.
- gt_hat[t] = number of singleton discovered modes after draws 0, ..., t,
  divided by t + 1.
- q[t] = probability that draw t + 1 does not match any mode discovered
  through t.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from failure_mode_embedding_utils import load_embeddings_cache  # noqa: E402


def load_verified_embeddings(
    raw_ids: list[str],
    texts: list[str],
    embeddings_cache_json: Path,
    embedding_model: str,
) -> np.ndarray:
    """Load cached embeddings and verify their model and source text."""
    cache = load_embeddings_cache(embeddings_cache_json)

    missing = [raw_id for raw_id in raw_ids if raw_id not in cache]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} embedding(s), e.g. {missing[0]!r}"
        )

    stale = [
        raw_id
        for raw_id, text in zip(raw_ids, texts, strict=True)
        if cache[raw_id]["model"] != embedding_model
        or cache[raw_id]["text"] != text
    ]
    if stale:
        raise ValueError(
            f"Found {len(stale)} stale embedding(s), e.g. {stale[0]!r}"
        )

    embeddings = np.asarray(
        [cache[raw_id]["embedding"] for raw_id in raw_ids],
        dtype=np.float64,
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cached embeddings must have nonzero norm")

    return embeddings / norms


def run_trial(
    bootstrap_session_indices: np.ndarray,
    session_embeddings: np.ndarray,
    similarity_threshold: float,
    n_population: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Online pairwise novelty under a growing catalog of all observations.

    The population has n_population sessions. Indices 0, ..., F - 1 (F =
    len(session_embeddings)) are sessions with a failure mode; indices
    F, ..., n_population - 1 are sessions without one (successes). A success
    draw is never novel and is not added to the catalog, and q divides by the
    whole population, so successes are probability mass that is never
    "missing".

    At index t:
    - z[t] = 1 iff draw t is not similar enough to any draw in 0, ..., t - 1.
    - q[t] = probability that draw t + 1 is not similar enough to any draw
      in 0, ..., t.
    - gt_hat[t] is a heuristic Good-Turing baseline. To define discrete modes
      for singleton counting, observations are separately assigned to an
      online greedy-cover taxonomy: each observation matches its most similar
      existing representative if above threshold, otherwise it starts a new
      mode. This taxonomy is used only for gt_hat, not for z or q.

    Every observed session is added to the pairwise catalog, regardless of
    whether it is novel.
    """
    n_sessions = len(session_embeddings)

    if n_population < n_sessions:
        raise ValueError(
            f"n_population ({n_population}) is smaller than the number of "
            f"failure sessions ({n_sessions})"
        )

    if np.any(bootstrap_session_indices < 0) or np.any(
        bootstrap_session_indices >= n_population
    ):
        raise ValueError("bootstrap_session_indices contains an invalid index")

    n_draws = len(bootstrap_session_indices)

    z = np.zeros(n_draws, dtype=np.int8)
    gt_hat = np.zeros(n_draws, dtype=np.float64)
    q = np.zeros(n_draws, dtype=np.float64)

    # Main pairwise novelty rule.
    covered = np.zeros(n_sessions, dtype=bool)

    # Auxiliary greedy taxonomy used only for the Good-Turing baseline.
    representatives: list[np.ndarray] = []
    mode_counts: list[int] = []
    n_singletons = 0

    for t, session_index in enumerate(bootstrap_session_indices):
        if session_index >= n_sessions:
            # Success draw: never novel, catalog and GT taxonomy unchanged.
            gt_hat[t] = n_singletons / (t + 1)
            q[t] = np.count_nonzero(~covered) / n_population
            continue

        embedding = session_embeddings[session_index]

        # Novel iff no previous observation covers the current session.
        z[t] = int(not covered[session_index])

        # Greedy-cover mode assignment for Good-Turing only.
        if not representatives:
            representatives.append(embedding)
            mode_counts.append(1)
            n_singletons += 1
        else:
            similarities = np.asarray(representatives) @ embedding
            best_mode = int(np.argmax(similarities))

            if similarities[best_mode] < similarity_threshold:
                representatives.append(embedding)
                mode_counts.append(1)
                n_singletons += 1
            else:
                if mode_counts[best_mode] == 1:
                    n_singletons -= 1
                mode_counts[best_mode] += 1

        gt_hat[t] = n_singletons / (t + 1)

        # Add every observation to the pairwise catalog.
        covered |= (
            session_embeddings @ embedding >= similarity_threshold
        )

        # True probability that the next bootstrap draw is novel.
        q[t] = np.count_nonzero(~covered) / n_population

    return z, gt_hat, q