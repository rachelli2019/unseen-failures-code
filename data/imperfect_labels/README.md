# Historical inputs for the final imperfect-label experiment

`input_snapshot/` contains the exact five files used for the reported 1,000-trial
comparison: empirical population IDs, saved Luna predictions, ground truth,
single-label failure-mode judgments, and the fixed historical taxonomy.
The experiment selects 4,961 sessions and 738 unsafe sessions; its taxonomy has
538 categories. Do not substitute a newer taxonomy when reproducing these results.

`input_hashes.json` records SHA-256 hashes of the input files. No new model calls are required. The label rule is implemented in `load_inputs` in [`src/run_imperfect_labels_reset.py`](../../src/run_imperfect_labels_reset.py); repeated draws count repeated ground-truth queries.
