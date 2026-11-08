#!/usr/bin/env python
"""Regenerate configs/protocol.json from the code that implements the protocol.

The configuration file is an export, not a second source of truth: every value
below is read from the module that enforces it, so the two cannot drift.
"""
from __future__ import annotations

import json
from pathlib import Path

from hrguard import core, measure


def main() -> int:
    cfg = {
        "_generated_by": "scripts/gen_configs.py",
        "_note": "Exported from the implementing modules; do not edit by hand.",
        "split": {
            "seed": core.SEED,
            "rule": "participants sorted by sha256(seed + participant_id), first 6 held out",
            "held_out": sorted(core.SEAL),
            "development": sorted(core.DEV),
            "n_subjects_total": len(core.SUBJECTS),
        },
        "windowing": {
            "window_ms": measure.WINDOW_MS,
            "stride_ms": measure.STRIDE_MS,
            "error_threshold_bpm": measure.ERROR_THRESHOLD_BPM,
            "min_good_samples": measure.MIN_GOOD_HR,
        },
        "clock_rule": {
            "candidate_offsets_ms": measure.POLAR_CANDIDATE_OFFSETS_MS,
            "min_best_overlap": measure.MIN_BEST_OVERLAP,
            "min_margin_over_second": measure.MIN_MARGIN_OVER_SECOND,
            "selection_uses_hr_values": False,
        },
        "statistics": {
            "alpha_one_sided": measure.ALPHA,
            "resampling_unit": "participant",
            "bootstrap": "percentile, participant-clustered",
            "default_n_boot": 8000,
            "seed": 20261104,
            "familywise_correction": "studentized max-T (Westfall-Young)",
        },
        "activity_groups": measure.ACTIVITY_GROUPS,
        "frozen_dev_reference": measure.DEV_REFERENCE,
        "frozen_dev_tolerance": measure.DEV_TOLERANCE,
    }
    out = Path(__file__).resolve().parents[1] / "configs" / "protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
