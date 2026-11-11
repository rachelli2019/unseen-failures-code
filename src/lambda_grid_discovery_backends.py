"""Discovery backends for run_lambda_grid_mixture.py.

Each backend answers one question: for a given trial RNG, what is the binary
novelty sequence z_t and the oracle unseen-failure probability q_t? Everything
downstream (the lambda grid, the restart mixture, the inversion) is identical
across backends, so the lambda-grid experiment is run on any setting by naming
its backend in the config's "discovery" block.

The z/q definitions are NOT reimplemented here -- each backend delegates to the
same module the corresponding headline pipeline uses, so a lambda-grid run and
the original run of that setting see the same trials for the same seed:

    fixed_primary    av_discovery_fixed_primary.run_bootstrap_trial   (run_primary_fixed.py)
    fixed_set_valued av_discovery_fixed_set_valued.run_trial          (run_set_valued_fixed.py)
    online_primary   av_discovery_online_primary.run_trial            (run_primary_online.py)
    bernoulli        i.i.d. Bernoulli(p), q_t = p exactly             (run_bernoulli_simulation.py)

A backend is a dict with:
    prepare(spec)             -> worker state (picklable; sent to every worker once)
    draw(state, rng, n_draws) -> (z int8[n_draws], q float64[n_draws])
    describe(spec, state)     -> JSON-able record of what the population was
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from failure_mode_embedding_utils import resolve_primary_records_texts  # noqa: E402

from av_core import require_file, require_probability  # noqa: E402
from av_discovery_fixed_primary import (  # noqa: E402
    build_session_population,
    load_session_to_failure_mode,
    run_bootstrap_trial,
)
from av_discovery_fixed_set_valued import build_population, load_session_to_failure_modes  # noqa: E402
from av_discovery_fixed_set_valued import run_trial as run_set_valued_trial  # noqa: E402
from av_discovery_online_primary import load_verified_embeddings  # noqa: E402
from av_discovery_online_primary import run_trial as run_online_trial  # noqa: E402
from evaluation_population import load_population_session_ids, require_within_population  # noqa: E402


def _require_number(spec: dict, key: str, backend: str) -> float:
    """A real number -- not a string that looks like one, and not a bool.

    bool subclasses int in Python, so an unguarded float() turns True into 1.0, and
    float("0.05") silently accepts a quoted number from JSON. Both are config errors.
    """
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"discovery backend {backend!r} key {key!r} must be a number; "
            f"got {type(value).__name__} {value!r}"
        )
    return float(value)


def _require_str(spec: dict, key: str, backend: str) -> str:
    value = spec[key]
    if not isinstance(value, str) or not value:
        raise TypeError(
            f"discovery backend {backend!r} key {key!r} must be a non-empty string; "
            f"got {type(value).__name__} {value!r}"
        )
    return value


def _validated_draw(z, q, n_draws: int, backend: str) -> tuple[np.ndarray, np.ndarray]:
    """Check a backend's (z, q) before it leaves this module.

    The runners check lengths only. A z that is not 0/1, or a q outside [0, 1], would
    otherwise flow into the e-process and come back out as a plausible-looking bound
    with nothing to distinguish it from a correct one. Cost is a few vectorised passes
    per trial against ~100 h x h array passes in the inversion, so it is free in context.
    """
    z = np.asarray(z)
    q = np.asarray(q, dtype=np.float64)
    if z.shape != (n_draws,) or q.shape != (n_draws,):
        raise ValueError(
            f"discovery backend {backend!r} returned z{z.shape} q{q.shape}; "
            f"expected both ({n_draws},)"
        )
    if not np.all((z == 0) | (z == 1)):
        offending = np.unique(z[(z != 0) & (z != 1)])[:5]
        raise ValueError(
            f"discovery backend {backend!r} returned a non-binary z; saw {offending!r}"
        )
    if not np.all(np.isfinite(q)):
        raise ValueError(f"discovery backend {backend!r} returned a non-finite q")
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError(
            f"discovery backend {backend!r} returned q outside [0, 1]; "
            f"range [{q.min()!r}, {q.max()!r}]"
        )
    return z.astype(np.int8), q


def _require_keys(spec: dict, backend: str, required: set[str]) -> None:
    """Exactly `required`, no missing keys and no unexpected extras (a typo must
    fail loudly rather than be ignored)."""
    keys = set(spec) - {"backend"}
    missing = required - keys
    if missing:
        raise ValueError(f"discovery backend {backend!r} is missing key(s): {sorted(missing)}")
    extra = keys - required
    if extra:
        raise ValueError(f"discovery backend {backend!r} has unexpected key(s): {sorted(extra)}")


# --------------------------------------------------------------------------
# fixed taxonomy, one failure mode per session (the primary representation)
# --------------------------------------------------------------------------

def _fixed_primary_prepare(spec: dict) -> dict:
    _require_keys(spec, "fixed_primary", {"taxonomy_json", "population_json"})
    taxonomy_json = Path(_require_str(spec, "taxonomy_json", "fixed_primary"))
    require_file(taxonomy_json, "discovery.taxonomy_json")
    population_json = Path(_require_str(spec, "population_json", "fixed_primary"))
    session_to_mode = load_session_to_failure_mode(taxonomy_json)
    mode_per_unit, mode_frequency, n_population = build_session_population(
        session_to_mode, load_population_session_ids(population_json)
    )
    return {"mode_per_unit": mode_per_unit, "mode_frequency": mode_frequency,
            "n_population": n_population}


def _fixed_primary_draw(state: dict, rng: np.random.Generator, n_draws: int):
    draw_indices = rng.integers(0, state["n_population"], size=n_draws)
    z, _gt_hat, q = run_bootstrap_trial(draw_indices, state["mode_per_unit"], state["mode_frequency"])
    return _validated_draw(z, q, n_draws, "fixed_primary")


def _fixed_primary_describe(spec: dict, state: dict) -> dict:
    return {"backend": "fixed_primary", "taxonomy_json": str(spec["taxonomy_json"]),
            "population_json": str(spec["population_json"]),
            "n_population": state["n_population"], "n_modes": len(state["mode_frequency"])}


# --------------------------------------------------------------------------
# fixed taxonomy, a SET of failure modes per session
# --------------------------------------------------------------------------

def _fixed_set_valued_prepare(spec: dict) -> dict:
    _require_keys(spec, "fixed_set_valued", {"taxonomy_json", "population_json"})
    taxonomy_json = Path(_require_str(spec, "taxonomy_json", "fixed_set_valued"))
    require_file(taxonomy_json, "discovery.taxonomy_json")
    population_json = Path(_require_str(spec, "population_json", "fixed_set_valued"))
    session_to_modes = load_session_to_failure_modes(taxonomy_json)
    population_session_ids = load_population_session_ids(population_json)
    require_within_population(session_to_modes, population_session_ids, str(taxonomy_json))
    modes_by_index, mode_to_indices = build_population(session_to_modes, population_session_ids)
    return {"modes_by_index": modes_by_index, "mode_to_indices": mode_to_indices,
            "n_population": len(modes_by_index)}


def _fixed_set_valued_draw(state: dict, rng: np.random.Generator, n_draws: int):
    draw_indices = rng.integers(0, state["n_population"], size=n_draws)
    z, q = run_set_valued_trial(draw_indices, state["modes_by_index"], state["mode_to_indices"])
    return _validated_draw(z, q, n_draws, "fixed_set_valued")


def _fixed_set_valued_describe(spec: dict, state: dict) -> dict:
    return {"backend": "fixed_set_valued", "taxonomy_json": str(spec["taxonomy_json"]),
            "population_json": str(spec["population_json"]),
            "n_population": state["n_population"], "n_modes": len(state["mode_to_indices"])}


# --------------------------------------------------------------------------
# online pairwise novelty over embeddings (no fixed taxonomy)
# --------------------------------------------------------------------------

def _online_primary_prepare(spec: dict) -> dict:
    _require_keys(spec, "online_primary",
                  {"records_json", "population_json", "embeddings_cache_json", "embedding_model",
                   "similarity_threshold"})
    records_json = Path(_require_str(spec, "records_json", "online_primary"))
    embeddings_cache_json = Path(_require_str(spec, "embeddings_cache_json", "online_primary"))
    embedding_model = _require_str(spec, "embedding_model", "online_primary")
    require_file(records_json, "discovery.records_json")
    require_file(embeddings_cache_json, "discovery.embeddings_cache_json")
    similarity_threshold = _require_number(spec, "similarity_threshold", "online_primary")
    require_probability(similarity_threshold, "discovery.similarity_threshold")
    population_json = Path(_require_str(spec, "population_json", "online_primary"))
    session_ids, texts = resolve_primary_records_texts(records_json)
    population_session_ids = load_population_session_ids(population_json)
    require_within_population(session_ids, population_session_ids, str(records_json))
    unit_embeddings = load_verified_embeddings(
        session_ids, texts, embeddings_cache_json, embedding_model
    )
    # Indices past the embedding rows are the population's sessions without a failure mode.
    return {"unit_embeddings": unit_embeddings, "similarity_threshold": similarity_threshold,
            "n_population": len(population_session_ids)}


def _online_primary_draw(state: dict, rng: np.random.Generator, n_draws: int):
    draw_indices = rng.integers(0, state["n_population"], size=n_draws)
    z, _gt_hat, q = run_online_trial(
        draw_indices, state["unit_embeddings"], state["similarity_threshold"], state["n_population"]
    )
    return _validated_draw(z, q, n_draws, "online_primary")


def _online_primary_describe(spec: dict, state: dict) -> dict:
    return {"backend": "online_primary", "records_json": str(spec["records_json"]),
            "population_json": str(spec["population_json"]),
            "embeddings_cache_json": str(spec["embeddings_cache_json"]),
            "embedding_model": spec["embedding_model"],
            "similarity_threshold": state["similarity_threshold"],
            "n_population": state["n_population"],
            "embedding_dim": int(state["unit_embeddings"].shape[1])}


# --------------------------------------------------------------------------
# i.i.d. Bernoulli(p): every z_t = 1 is a brand-new mode, so q_t = p exactly
# --------------------------------------------------------------------------

def _bernoulli_prepare(spec: dict) -> dict:
    _require_keys(spec, "bernoulli", {"p"})
    p = _require_number(spec, "p", "bernoulli")
    require_probability(p, "discovery.p")
    return {"p": p}


def _bernoulli_draw(state: dict, rng: np.random.Generator, n_draws: int):
    p = state["p"]
    z = (rng.random(n_draws) < p).astype(np.int8)
    return _validated_draw(z, np.full(n_draws, p, dtype=np.float64), n_draws, "bernoulli")


def _bernoulli_describe(spec: dict, state: dict) -> dict:
    return {"backend": "bernoulli", "p": state["p"], "n_population": None}


BACKENDS = {
    "fixed_primary": {"prepare": _fixed_primary_prepare, "draw": _fixed_primary_draw,
                      "describe": _fixed_primary_describe},
    "fixed_set_valued": {"prepare": _fixed_set_valued_prepare, "draw": _fixed_set_valued_draw,
                         "describe": _fixed_set_valued_describe},
    "online_primary": {"prepare": _online_primary_prepare, "draw": _online_primary_draw,
                       "describe": _online_primary_describe},
    "bernoulli": {"prepare": _bernoulli_prepare, "draw": _bernoulli_draw,
                  "describe": _bernoulli_describe},
}


def get_backend(spec: dict) -> dict:
    if not isinstance(spec, dict) or "backend" not in spec:
        raise ValueError(f"discovery must be an object with a 'backend' key; got {spec!r}")
    name = spec["backend"]
    if name not in BACKENDS:
        raise ValueError(f"unknown discovery backend {name!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[name]
