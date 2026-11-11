# Experiment inputs

| Directory | Contents |
|---|---|
| `population/` | Evaluation population identifiers |
| `taxonomy/` | Fixed primary, set-valued, and LLM-merged taxonomies |
| `failure_modes/` | Per-session failure labels and supporting evidence |
| `embeddings/` | Cached embeddings and records for categorization |
| `imperfect_labels/` | Fixed population, taxonomy, ground-truth outcomes, and predictions for the imperfect-label comparison |
| `processed/` | Session data and development/test splits |

The runners read these inputs using paths in `configs/`. Prompts used for failure categorization are in [`prompts/`](../prompts/).
