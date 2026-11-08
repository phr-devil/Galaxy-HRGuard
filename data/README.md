# Data

No data are stored in this repository. The measurement runs on a public dataset
that you obtain yourself, and the participant-level tables it produces stay on
your machine.

## Upstream dataset

| Field | Value |
| --- | --- |
| Name | GalaxyPPG: A PPG Signal Dataset Collected in Semi-Naturalistic Settings Using Galaxy Watch |
| Version | 1.0.0 |
| DOI | `10.5281/zenodo.14635823` |
| Record | https://zenodo.org/records/14635823 |
| Licence | CC BY 4.0 |
| Archive | `GalaxyPPG.zip`, 481,594,233 bytes, `md5:d9a48bfcd07928fb22b96dc40cf2e7d4` |
| Authors' code | https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code |

The recorded MD5 is the one the frozen analysis used. If the archive you obtain
has a different checksum, the upstream record changed and the frozen counts will
not necessarily reproduce.

CC BY 4.0 permits redistribution with attribution. This repository still does
not redistribute the files, for two reasons: the archive is large, and the
licence obligation would then fall on every downstream copy of this repository
rather than on the dataset record itself. Cite the dataset instead.

## What the measurement reads

Per participant, five files. Only four enter the estimands; the accelerometer
file is read by the legal feature table but not by the headline ratio.

```
P02/GalaxyWatch/PPG.csv      green-channel PPG          (legal input)
P02/GalaxyWatch/HR.csv       device-reported HR + status (legal input)
P02/GalaxyWatch/ACC.csv      wrist accelerometer        (legal input)
P02/PolarH10/HR.csv          chest-belt reference HR    (reference only)
P02/Event.csv                activity protocol timeline  (used to define sessions)
```

Two rules keep the reference out of the prediction path, and both are enforced
in code rather than by convention:

1. The clock offset between the device and the reference is chosen from
   **timestamp ranges only**, over a closed candidate set, and the reference HR
   values are never consulted when choosing it.
2. Any window whose reference value is missing is excluded from the ratio's
   denominator, so a window can never be scored against a reference that is not
   there.

## Why the raw files are not committed

Two classes of file are excluded from version control:

- **Raw upstream files.** Unmodified copies of a third-party dataset.
- **Participant-level derived tables.** The window tables carry a heart-rate
  value for every 8 s window of every participant. They are re-identifiable in
  the sense that matters: they are the raw measurement at a finer grain than the
  published aggregates, and nobody needs them to check the paper.

The committed artefacts in `evidence/` are aggregates: counts, ratios and
interval bounds, with no per-window values.

## Held-out participants

The split is frozen in `configs/cohort.json` and was fixed by hashing a public
seed against participant identifiers, before any label was opened. The held-out
folders are not distributed here. If you have been granted access to them, put
them under `data/sealed` (or `$HRGUARD_SEALED`) in the same per-participant
layout, and `scripts/12_verify_result.py` will recompute the held-out ratio.
That script refuses to run without `HRGUARD_SEALED_AUTHORIZED=1`.
