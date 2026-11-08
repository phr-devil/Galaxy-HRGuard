# Galaxy-HRGuard

Reproduction code for a three-state reliability measurement of wrist heart rate
on a public semi-naturalistic wearable dataset.

The measurement answers a question that accuracy-only reporting cannot answer.
When a wrist device reports a heart rate, the report can be correct, it can be
wrong by more than 10 bpm, or the device can decline to report at all. Most
studies score accuracy over the values that were delivered, which silently
deletes the third state. This repository implements the alternative: a
three-state label, a denominator that keeps refused windows in the population,
and a ratio that states how unreliable windows split between wrong output and
no output.

The headline quantity is

```
E1 = n_wrong / (n_wrong + n_no-output)
```

computed on windows where the device and the reference both had an opportunity
to report. On the held-out cohort this ratio is **0.1212** (147 of 1,213
unreliable windows, 95% participant-clustered bootstrap interval
[0.0493, 0.2105]); the remaining 0.8788 of unreliable windows carry no output at
all. Reporting accuracy over delivered values therefore describes a different
population from the one a monitoring claim is about.

## What is in the repository

| Path | Contents |
| --- | --- |
| `src/hrguard/` | The measurement protocol as an importable package. `raw` audits the licensed files and decides the clock rule, `windows` builds 8 s windows with a 2 s stride and assigns the three-state label, `signals` resamples the legal channels, `endpoint` attaches reference values, `features` computes the legal feature table, `measure` computes the frozen estimands with the cluster bootstrap and the decision criteria, `online` implements the online alerting protocol, and `composition`, `blocks`, `within`, `outage` implement the exploratory analyses reported in Table 3 of the manuscript. |
| `scripts/` | One thin command-line entry point per stage, numbered in execution order. |
| `tests/` | 26 unit tests covering the label arithmetic, the tie-aware risk-coverage estimator, the subject-macro metrics, the clock rule and the ZIP intake path. No network, no licensed data, synthetic fixtures only. |
| `configs/` | The frozen protocol constants and the cohort split, generated from the code that implements them. |
| `evidence/` | Committed aggregate artefacts of the frozen analysis: the confirmation ledger, the claim freeze, the independent recomputation and the assembled result record. |
| `data/` | Instructions for obtaining the licensed dataset. No data are stored in this repository. |

## Install

Python 3.12 or newer is required by the pinned scientific stack. The pinned
environment is the one that produced the frozen artefacts:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[test]"
python -m pytest -q
```

`pip install -e .` puts `hrguard` on the path, so `scripts/` can be run
directly. Without installing, prefix the commands with `PYTHONPATH=src`.

## Data

The dataset is **not** redistributed here. It is GalaxyPPG v1.0.0, published
under CC BY 4.0 at Zenodo record
[14635823](https://zenodo.org/records/14635823) (DOI `10.5281/zenodo.14635823`),
with the authors' own preprocessing code at
[Kaist-ICLab/GalaxyPPG-Supplementary-Code](https://github.com/Kaist-ICLab/GalaxyPPG-Supplementary-Code).
Check the upstream terms before any redistribution of your own.

Download the archive, then either extract the participant folders yourself or
let the intake step do it:

```bash
python -c "from hrguard.prepare import unpack_dev; \
    unpack_dev('path/to/GalaxyPPG.zip', 'data/raw', 'outputs/zip_manifest.json')"
```

Only the development participants are extracted, five files each; the held-out
participants stay inside the archive unless you open them deliberately.

The layout expected under `data/raw` (or `$HRGUARD_DATA`) is one folder per
participant:

```
data/raw/P02/GalaxyWatch/PPG.csv
data/raw/P02/GalaxyWatch/HR.csv
data/raw/P02/GalaxyWatch/ACC.csv
data/raw/P02/PolarH10/HR.csv
data/raw/P02/Event.csv
```

The participant split is frozen in `configs/cohort.json`: 18 development
participants and 6 held-out participants, assigned by a published hash rule that
never looked at a label. Participants whose files are absent are excluded by the
source audit rather than silently dropped downstream.

## Running the measurement

Every command below reads only from the paths you pass and writes only into
`outputs/` or the `--out` you name. `$RAW` is the raw data root; `$OUT` is
`outputs/evidence`.

```bash
export HRGUARD_DATA=/path/to/data/raw
OUT=outputs/evidence

# 1. Source audit: file presence, clock rule, evaluability
python scripts/01_audit_raw.py --raw "$HRGUARD_DATA" --out $OUT/raw_audit_dev.json

# 2. Window audit: 8 s windows, 2 s stride, three-state label
python scripts/02_probe_windows.py --raw "$HRGUARD_DATA" \
    --audit $OUT/raw_audit_dev.json --out $OUT/window_probe_dev.json

# 3. Uniform resampling of the legal channels
python scripts/03_extract_signals.py --raw "$HRGUARD_DATA" \
    --audit $OUT/raw_audit_dev.json --probe $OUT/window_probe_dev.json \
    --out $OUT/dev_features.csv

# 4. Window universe with reference values attached
python scripts/04_build_universe.py --raw "$HRGUARD_DATA" \
    --audit $OUT/raw_audit_dev.json --out $OUT/pivot/endpoint_audit.json
# writes $OUT/pivot/endpoint_audit.json (report) and
#        $OUT/pivot/endpoint_audit_windows.csv (the window table)

# 5. Legal feature table
python scripts/05_extract_features.py --raw "$HRGUARD_DATA" \
    --audit $OUT/raw_audit_dev.json \
    --universe $OUT/pivot/endpoint_audit_windows.csv \
    --old-features $OUT/dev_features.csv \
    --out $OUT/pivot/pivot_features.csv --receipt $OUT/feature_receipt.json

# 6. Frozen estimands, cluster bootstrap and criteria.
#    The dev-verify stage asserts the frozen reference values and exits non-zero
#    if any of them moves.
python scripts/06_dev_verify.py --stage dev-verify --raw "$HRGUARD_DATA"
```

The exploratory analyses behind Table 3 of the manuscript run on the same raw
root:

```bash
python scripts/07_gap_imputation.py     # no-output gap structure, imputation error
python scripts/08_motion_blocks.py      # coverage loss under high motion
python scripts/09_null_contrast.py      # equal-budget null contrast
python scripts/10_outage_peak.py        # peak underestimation inside gaps
```

## Expected outputs

| Command | Output | What to check |
| --- | --- | --- |
| `01` | `outputs/evidence/raw_audit_dev.json` | `summary.state_counts` — participants with a missing required file are marked, not dropped |
| `02` | `outputs/evidence/window_probe_dev.json` | `bad10` and `watch_ok` are the numerator and denominator of the wrong-output count |
| `04` | `outputs/evidence/pivot/endpoint_audit_windows.csv` | on the development cohort: 24,924 windows, 726 wrong, 2,723 no-output, ratio 0.210496 |
| `06` | `outputs/evidence/v1_6/v1_6_dev_verify.json` | `state == "DEV_VERIFY_OK"`; `layer1_estimand_layer.all_pass == true`; `layer2_window_layer.mismatches[*].n_differing == 0` |
| `11` | `outputs/V1_7_CONFIRMATION_RESULT.json` | reproduces the committed record byte for byte from committed artefacts |
| `12` | `outputs/evidence/v1_7/v1_7_independent_verification.json` | independent recomputation of the same ratio from the held-out files, when you have authorised access to them |

The reproduction check that needs no data at all is the assembly step: it
re-derives the headline record from the committed artefacts in `evidence/`.

```bash
python scripts/11_build_result.py
diff <(python -c "import json;print(json.dumps(json.load(open('outputs/V1_7_CONFIRMATION_RESULT.json')),sort_keys=True,indent=1))") \
     <(python -c "import json;print(json.dumps(json.load(open('evidence/V1_7_CONFIRMATION_RESULT.json')),sort_keys=True,indent=1))")
```

## What this code does and does not reproduce

Reproducible from the licensed data with the commands above: the three-state
label, the window universe, the legal feature table, the frozen estimands (E1
and its interval, the subject-equal variant, the activity gradient, the
between-participant spread, the refusal rate, the variance decomposition) and
the exploratory analyses of Table 3.

The held-out cohort is a **single-shot** protocol. Its counts are committed in
`evidence/`, and `scripts/12_verify_result.py` re-derives them from the held-out
files, but the repository will not re-run the confirmation: the script refuses to
proceed unless `HRGUARD_SEALED_AUTHORIZED=1` is set, and the held-out files are
not distributed here. Re-running a one-shot confirmation on the same inputs
would not be a confirmation.

Not included: the five-arm offline ranking pilot and the online alerting audit
that together produce the AUROC ladder and the recall figures discussed in the
manuscript. Those analyses depend on per-window out-of-fold prediction tables
that are participant-level and are therefore not redistributable. The protocol
they audit is implemented in `hrguard.online`, so the definitions are available
even though the audit inputs are not.

Where the manuscript and this code disagree, the manuscript is the record of
what was done and this code is the instrument; a disagreement is a bug in the
instrument and should be reported as one.

## Licence

No open-source licence has been granted yet. The copyright holder has not chosen
one, so this repository is published for inspection and review and all rights
are reserved; see `LICENSE`. The upstream dataset is a separate work under
CC BY 4.0 and is not covered by that statement.

## Citation

`CITATION.cff` describes this software. The manuscript is not published yet, so
no article DOI is asserted anywhere in this repository. If you use the method,
cite the dataset and this repository:

> Galaxy-HRGuard v0.1.0. Three-state reliability measurement for wrist heart
> rate. https://github.com/phr-devil/Galaxy-HRGuard

## Verification status

Checked on 2026-11-08 with the pinned environment on a clean virtual environment
(Python 3.13.14, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1).

| Check | Result |
| --- | --- |
| `pip install -e ".[test]"` in a fresh venv | succeeds |
| `pytest -q` | 26 passed |
| Import of every module | succeeds |
| Participant split regenerated from the published seed | reproduces the frozen held-out set |
| Window universe rebuilt from the raw development files | 24,924 windows, 726 wrong, 2,723 no-output |
| `06_dev_verify.py --stage dev-verify` | `DEV_VERIFY_OK` |
| Window layer rebuilt and compared row by row | 24,924 rows × 5 columns, 0 differences |
| Frozen development estimands recomputed | all 8 agree to `0.00e+00` |
| Headline record reassembled from committed artefacts | identical to the committed record |

The development estimands that reproduce exactly are the wrong-output share
(0.2104957959), its subject-equal variant (0.1905505978), the within-participant
activity gradient (3.6473391616), the quartile ratio of participant error rates
(3.8260548995), the refusal rate (0.1086692670), and the variance decomposition
(0.4327 for participant, 0.1283 for intensity class, 0.4390 for the interaction).

`v0.1.0` remains a research code release rather than a validated reproduction
package: the held-out confirmation is not re-run here by design, and the offline
ranking and online alerting audits are not included.
