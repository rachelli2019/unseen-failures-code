"""Shared config-file loading for the run_*.py pipeline scripts.

Each run's exact parameters live in one checked-in JSON file (see configs/)
instead of a long CLI flag list, so a run is fully reproducible from that
file alone -- rerun the same config, get the same run.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_run_config(config_path: Path, required_keys: set[str], path_keys: set[str]) -> dict:
    """Load a JSON run config and validate it has exactly required_keys --
    no missing keys, and no unexpected extras (a typo'd key must fail loudly,
    not be silently ignored). Values for path_keys are converted to Path.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"--config does not exist: {config_path}")

    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")

    missing = required_keys - set(config)
    if missing:
        raise ValueError(f"{config_path} is missing required key(s): {sorted(missing)}")
    extra = set(config) - required_keys
    if extra:
        raise ValueError(f"{config_path} has unexpected key(s): {sorted(extra)}")

    for key in path_keys:
        config[key] = Path(config[key])
    return config
