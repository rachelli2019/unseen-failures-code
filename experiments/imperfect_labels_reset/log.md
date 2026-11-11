# Integration log

2026-11-10: Ported the final perfect/IPW B-PAC/PPI r=0 comparison to public-release.
Retained 1,000-trial results and completed PPI stopping costs. Added pointwise
95% trial percentile shading for bounds and percentile error bars for label
counts. Renamed the cost panel to Ground-truth labels used. No simulation rerun.

Validation: all 9 unit tests passed; all five input SHA-256 hashes match the original run. A 100-draw seed-0 replay exactly matches all 75 stored trajectory fields. Both PNG/PDF figures were regenerated; the three-panel figure was visually checked. The coverage legend clears the nominal line.
