# Historical inputs for the final imperfect-label experiment

`input_snapshot/` contains the exact five files used for the reported 1,000-trial
comparison: empirical population IDs, saved Luna predictions, ground truth,
single-label failure-mode judgments, and the fixed historical taxonomy.
The experiment selects 4,961 sessions and 738 unsafe sessions; its taxonomy has
538 categories. Do not substitute a newer taxonomy when reproducing these results.

Files were recovered from revision `10502f7` during the original experiment and
checked against the original run's input hashes. `input_hashes.json` records their
SHA-256 hashes. The saved simulation metadata retains its original snapshot path;
the release configs point to this directory. No new model calls are required.

See `experiments/imperfect_labels_reset/spec.md` for the truth definition and
categorizer assumptions. Repeated draws count repeated ground-truth queries.
