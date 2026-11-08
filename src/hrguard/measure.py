"""v1.6 -- THE ONE-SHOT CONFIRMATION RUNNER (measurement only, zero fitting).

This is the single code path that will be used BOTH to
  (a) verify itself on DEV (stage `dev-verify`), reproducing the frozen DEV
      numbers bit-for-bit, and
  (b) run the one-shot confirmation on the SEALED cohort (stage `confirm`).

Nothing in here fits a model, calibrates a threshold, searches a
hyper-parameter, or reads a hypothesis. It measures.

STAGES
------
  dev-verify : runs the identical computation over the DEV cohort and asserts
                the frozen DEV reference values. Reads only DEV data. Never
                touches SEALED. This is the G0.4 gate.
  preflight  : runs ONLY the frozen clock rule over the SEALED participants
                and writes the alignment diagnostics to disk BEFORE any label
                is computed. Produces no estimand.
  confirm    : runs the full one-shot measurement and writes the estimands.

AUTHORISATION GATE
------------------
Stages `preflight` and `confirm` refuse to run unless the environment
variable HRGUARD_SEALED_AUTHORIZED is exactly "1". The authorisation is
one-directional and single-use: the run writes a ledger entry and refuses a
second confirmation run.

PROHIBITIONS ENFORCED IN CODE
-----------------------------
  * the clock decision uses TIMESTAMPS ONLY (closed candidate set, overlap
    fraction); reference-HR error is never consulted;
  * a participant failing any governance clause is DATA_NOT_EVALUABLE and
    enters neither a numerator nor a denominator;
  * no threshold, group, estimand or inclusion rule may be changed after the
    SEALED files are opened -- the frozen values live in ONE place and are
    asserted against V1_5_CLAIM_FREEZE.json before the run.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import paths as _paths
from .raw import FILES, read_minimal
from .windows import (MIN_GOOD_HR, clean_series,
                      sessions_from_events, window_slice)

HERE = _paths.output_dir()
OUT = HERE / "evidence" / "v1_6"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- constants
# FROZEN. Mirrors audit_raw_dev.py verbatim; asserted below against the
# DEV audit evidence so that a drift in either place is caught.
POLAR_CANDIDATE_OFFSETS_MS = {"raw": 0, "author_minus_9h": -32400000}
MIN_BEST_OVERLAP = 0.75
MIN_MARGIN_OVER_SECOND = 0.30
PASS_RES_LO_S = (-578.8, 109.1)
PASS_RES_HI_S = (-509.4, 130.9)

WINDOW_MS = 8000
STRIDE_MS = 2000
ERROR_THRESHOLD_BPM = 10.0
ALPHA = 0.05          # one-sided, pre-registered; used by the bootstrap tests

ACTIVITY_GROUPS = {
    "G1_sedentary_desk": ["adaptation", "baseline", "meditation-1", "meditation-2",
                          "screen-reading", "keyboard-typing", "mobile-typing"],
    "G2_speech": ["tsst-prep", "tsst-speech", "ssst-prep", "ssst-sing"],
    "G3_rest": ["rest-1", "rest-2", "rest-3", "rest-4", "rest-5"],
    "G4_active": ["standing", "walking", "jogging", "running"],
}
PURE_ACTIVE = ["walking", "jogging", "running"]
CLEAN_QUIET = ["baseline", "adaptation", "meditation-1", "meditation-2",
               "screen-reading", "mobile-typing"]

OUR_SEALED = ["P08", "P13", "P17", "P19", "P20", "P23"]

# The five raw files required by the frozen governance rule G-EVAL.
REQUIRED_FILES = ["GalaxyWatch/HR.csv", "PolarH10/HR.csv", "Event.csv",
                  "GalaxyWatch/PPG.csv", "GalaxyWatch/ACC.csv"]
# Files actually needed by the estimands (ACC is not one of them).
ESTIMAND_FILES = ["GalaxyWatch/HR.csv", "PolarH10/HR.csv", "Event.csv",
                  "GalaxyWatch/PPG.csv"]

# Frozen DEV reference values (V1_5_CLAIM_FREEZE.json / V1_5_SEALED_RUNBOOK.md
# step 3). The dev-verify stage asserts these to 1e-12.
DEV_REFERENCE = {
    "pooled_supply_share": 0.2104957958828646,
    "subject_equal_supply_share": 0.1905505977934112,
    "within_person_activity_gradient": 3.6473391616266513,
    "error_rate_q75_over_q25": 3.8260548995033163,
    "mean_refusal_rate": 0.1086692669815535,
    "eta2_participant": 0.4327028871941862,
    "eta2_group": 0.128277953151707,
    "eta2_cell": 0.4390191596541068,
}
DEV_TOLERANCE = 1e-9


# ------------------------------------------------------------------ helpers
def overlap_fraction(test, reference):
    """Fraction of TEST timestamps that fall inside the REFERENCE range.

    Verbatim semantics of audit_raw_dev.overlap_fraction: TIMESTAMPS ONLY.
    """
    test = np.sort(np.asarray(test, dtype=float))
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(test) == 0 or len(reference) == 0:
        return 0.0
    lo, hi = reference[0], reference[-1]
    return float(np.mean((test >= lo) & (test <= hi)))


def infer_offset(reference_times, watch_times):
    """Frozen clock decision. TIMESTAMPS ONLY -- never uses HR error.

    Verbatim semantics of audit_raw_dev.infer_offset.
    """
    score = {k: overlap_fraction(np.asarray(reference_times, dtype=float) + v,
                                 np.asarray(watch_times, dtype=float))
             for k, v in POLAR_CANDIDATE_OFFSETS_MS.items()}
    ranks = sorted(score, key=score.get, reverse=True)
    best, second = ranks[:2]
    ok = score[best] >= MIN_BEST_OVERLAP and (score[best] - score[second]) >= MIN_MARGIN_OVER_SECOND
    return {
        "candidate_overlap_fraction": score,
        "selected_offset_ms": POLAR_CANDIDATE_OFFSETS_MS[best] if ok else None,
        "selected_recipe": best if ok else None,
        "status": ("UNIQUE_TIME_RANGE_MATCH" if ok else
                   ("AMBIGUOUS" if score[best] >= MIN_BEST_OVERLAP else "NO_MATCH")),
        "rule": "best >= 0.75 AND (best - second) >= 0.30, timestamp overlap only",
    }


def clock_guard(polar_lo, polar_hi, watch_hr_lo, watch_hr_hi, offset_ms):
    """The pre-declared continuous envelope guard on the chosen offset."""
    if offset_ms is None:
        return {"passed": False, "reason": "no offset selected"}
    res_lo = (polar_lo + offset_ms - watch_hr_lo) / 1000.0
    res_hi = (polar_hi + offset_ms - watch_hr_hi) / 1000.0
    ok = (PASS_RES_LO_S[0] <= res_lo <= PASS_RES_LO_S[1] and
          PASS_RES_HI_S[0] <= res_hi <= PASS_RES_HI_S[1])
    return {
        "res_lo_s": float(res_lo), "res_hi_s": float(res_hi),
        "pass_window_res_lo_s": list(PASS_RES_LO_S),
        "pass_window_res_hi_s": list(PASS_RES_HI_S),
        "passed": bool(ok),
        "reason": "" if ok else "residual envelope violated",
    }


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_safe(o):
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# --------------------------------------------------------- window labelling
def label_participant(raw_dir: Path, sid: str, offset_ms: int):
    """Build the frozen window table for one participant.

    Window grid, MIN_GOOD_HR rule and the >10 bpm error rule are identical to
    audit_pivot_endpoint.build_universe / extract_pivot_features.build.
    """
    sources = {}
    for name, (rel, cols) in FILES.items():
        src = raw_dir / sid / rel
        sources[name] = read_minimal(src, cols)

    ppg = clean_series(sources["watch_ppg"], "timestamp", ["ppg", "status"])
    wh = clean_series(sources["watch_hr"], "timestamp", ["hr", "hrStatus"])
    ref = clean_series(sources["polar_hr"].rename(columns={"phoneTimestamp": "timestamp"}),
                       "timestamp", ["hr"])
    ref["timestamp"] += offset_ms

    rows = []
    for session, s0, s1 in sessions_from_events(sources["events"]):
        for t0 in range(s0, s1 - WINDOW_MS + 1, STRIDE_MS):
            t1 = t0 + WINDOW_MS
            r = window_slice(ref, t0, t1)
            rv = pd.to_numeric(r.hr, errors="coerce").to_numpy(dtype=float)
            rv = rv[np.isfinite(rv) & (rv >= 40) & (rv <= 200)]
            h = window_slice(wh, t0, t1)
            hv = h.loc[(pd.to_numeric(h.hrStatus, errors="coerce") == 1)
                       & pd.to_numeric(h.hr, errors="coerce").between(40, 200), "hr"]
            hv = pd.to_numeric(hv, errors="coerce").to_numpy(dtype=float)
            ref_ok = len(rv) >= MIN_GOOD_HR
            watch_ok = len(hv) >= MIN_GOOD_HR
            err = (float(abs(np.median(hv) - np.mean(rv)))
                   if (ref_ok and watch_ok) else np.nan)
            rows.append({
                "participant": sid, "session": session, "start_ms": t0, "end_ms": t1,
                "ref_ok": bool(ref_ok), "watch_ok": bool(watch_ok),
                "n_watch_any": int(len(h)), "n_watch_valid": int(len(hv)),
                "ref_median": float(np.mean(rv)) if ref_ok else np.nan,
                "bad10": bool(np.isfinite(err) and err > ERROR_THRESHOLD_BPM),
            })
    df = pd.DataFrame(rows)
    df["watch_missing"] = df.ref_ok & ~df.watch_ok
    df["group"] = df["session"].map(
        {s: g for g, m in ACTIVITY_GROUPS.items() for s in m})
    return df, sources


# ------------------------------------------------------------- the estimands
def compute_estimands(df: pd.DataFrame) -> dict:
    """The seven frozen estimands, computed identically for DEV and SEALED."""
    df = df.copy()
    df["bad"] = df["bad10"] == True
    participants = sorted(set(df["participant"]))
    A = df[df.watch_ok == True]

    per = {}
    for p in participants:
        m_all = df[df.participant == p]
        m_A = m_all[m_all.watch_ok == True]
        w = int(m_A["bad"].sum())
        r = int(m_all["watch_missing"].sum())   # refused = ref_ok & ~watch_ok
        u = w + r
        per[p] = {
            "n_windows": int(len(m_all)),
            "n_reference_valid": int(m_all["ref_ok"].sum()),
            "n_A": int(len(m_A)),
            "n_wrong": w, "n_refused": r, "n_unreliable": u,
            "supply_share": (w / u) if u else float("nan"),
            "error_rate_A": float(m_A["bad"].mean()) if len(m_A) else float("nan"),
            "refusal_rate_all": float(m_all["watch_missing"].mean()) if len(m_all) else float("nan"),
        }

    # E1 -- pooled supply share
    n_wrong = int(A["bad"].sum())
    n_refuse = int(df["watch_missing"].sum())
    e1 = (n_wrong / (n_wrong + n_refuse)) if (n_wrong + n_refuse) else float("nan")

    # E1b -- subject-equal supply share
    e1b = float(np.mean([per[p]["supply_share"] for p in participants
                         if np.isfinite(per[p]["supply_share"])]))

    # E4 -- device refusal rate (participant-level mean of per-participant rates)
    e4 = float(np.mean([per[p]["refusal_rate_all"] for p in participants]))

    # E2 -- within-person activity gradient (paired log-odds)
    def paired_logodds_delta(p, hi_set, lo_set):
        m = df[df.participant == p]
        mA = m[m.watch_ok == True]
        a = mA[mA.session.isin(hi_set)]
        b = mA[mA.session.isin(lo_set)]
        if len(a) == 0 or len(b) == 0:
            return float("nan")
        la = np.log((a["bad"].sum() + 0.5) / (len(a) - a["bad"].sum() + 0.5))
        lb = np.log((b["bad"].sum() + 0.5) / (len(b) - b["bad"].sum() + 0.5))
        return float(la - lb)

    deltas = {p: paired_logodds_delta(p, PURE_ACTIVE, CLEAN_QUIET) for p in participants}
    dv = np.array([v for v in deltas.values() if np.isfinite(v)], float)
    e2 = float(dv.mean()) if len(dv) else float("nan")
    e2b = float((dv > 0).mean()) if len(dv) else float("nan")

    # E3 -- between-person spread (DESCRIPTIVE ONLY)
    rates = np.array([per[p]["error_rate_A"] for p in participants], float)
    q25, q75 = np.quantile(rates, [0.25, 0.75])
    e3 = float(q75 / q25) if q25 > 0 else float("nan")

    # E5 -- unit-weighted eta^2 (DESCRIPTIVE ONLY in v1.5)
    e5 = unit_weighted_eta2(df)

    return {
        "n_evaluable": len(participants),
        "participants": participants,
        "per_participant": per,
        "E1_pooled_supply_share": e1,
        "E1_pooled_supply_share_raw": {
            "n_wrong_value_windows": n_wrong,
            "n_refused_windows": n_refuse,
            "n_unreliable_windows": n_wrong + n_refuse,
            "numerator_definition": "windows where the device reported a value "
                                    "that differs from the Polar reference by "
                                    f"> {ERROR_THRESHOLD_BPM:.0f} bpm",
            "denominator_definition": "all unreliable reference-valid windows, "
                                       "i.e. wrong-value plus refused",
        },
        "E1b_subject_equal_supply_share": e1b,
        "E2_within_person_activity_gradient": e2,
        "E2_within_person_activity_gradient_raw": {
            "per_participant": deltas,
            "contrast": "log-odds(pure motion: walking/jogging/running) - "
                        "log-odds(quiet: baseline/adaptation/meditation-1/"
                        "meditation-2/screen-reading/mobile-typing)",
            "caveat": "the protocol order is FIXED for every participant, so "
                      "activity type and protocol position cannot be "
                      "separated; this is a within-participant conditional "
                      "contrast, NOT a causal effect of activity",
        },
        "E2b_positive_gradient_fraction": e2b,
        "E3_between_person_spread": e3,
        "E3_raw": {"per_participant_error_rate": {p: per[p]["error_rate_A"]
                                                  for p in participants},
                   "q25": float(q25), "q75": float(q75),
                   "tier": "DESCRIPTIVE ONLY -- no quantitative heterogeneity "
                           "claim is made from SEALED"},
        "E4_device_refusal_rate": e4,
        "E4_raw": {"per_participant_refusal_rate": {p: per[p]["refusal_rate_all"]
                                                     for p in participants}},
        "E5_eta2_participant_share": e5.get("eta2_participant"),
        "E5_detail": e5,
    }


def unit_weighted_eta2(df: pd.DataFrame) -> dict:
    """Classical BALANCED unit-weighted two-way decomposition (v1.5 stage A).

    IDENTICAL construction to v1_5_eta2_audit.anova(weight='unit'):
      1. restrict to population A (watch_ok == True);
      2. aggregate to CELLS of participant x activity group, with
         k = number of wrong-value windows and n = number of windows;
      3. cell log-odds  lo = log((k + 0.5) / (n - k + 0.5));
      4. ordinary two-way ANOVA on the cell log-odds with UNIT weights
         (every cell counts once), which for the balanced layout is
         orthogonal by construction: the three shares sum to exactly 1.
    """
    d = df[df.watch_ok == True].copy()
    d["bad"] = d["bad10"] == True
    d = d[d["group"].notna()]
    cells = (d.groupby(["participant", "group"])["bad"]
               .agg(k="sum", n="size").reset_index())
    cells = cells[cells["n"] > 0].copy()
    if cells.empty:
        return {"eta2_participant": float("nan"), "eta2_group": float("nan"),
                "eta2_cell": float("nan"), "sum": float("nan"),
                "n_cells": 0}
    lo = np.log((cells["k"].to_numpy(float) + 0.5)
                / (cells["n"].to_numpy(float) - cells["k"].to_numpy(float) + 0.5))
    part = cells["participant"].to_numpy()
    grp = cells["group"].to_numpy()
    grand = float(lo.mean())
    pm = pd.Series(lo).groupby(pd.Series(part)).mean()
    gm = pd.Series(lo).groupby(pd.Series(grp)).mean()
    dp = pd.Series(part).map(pm - grand).to_numpy(float)
    dg = pd.Series(grp).map(gm - grand).to_numpy(float)
    ss_p = float((dp ** 2).sum())
    ss_g = float((dg ** 2).sum())
    ss_c = float(((lo - (grand + dp + dg)) ** 2).sum())
    tot = ss_p + ss_g + ss_c
    if tot <= 0:
        return {"eta2_participant": float("nan"), "eta2_group": float("nan"),
                "eta2_cell": float("nan"), "sum": float("nan"),
                "n_cells": int(len(cells))}
    return {"eta2_participant": ss_p / tot, "eta2_group": ss_g / tot,
            "eta2_cell": ss_c / tot, "sum": (ss_p + ss_g + ss_c) / tot,
            "orthogonal": True, "n_cells": int(len(cells)),
            "construction": "cell-level (participant x activity group) k/n "
                             "aggregation, Laplace-smoothed log-odds, "
                             "unit-weighted ordinary two-way ANOVA",
            "tier": "DESCRIPTIVE ONLY -- the cell term mixes interaction with "
                    "binomial sampling error (one observation per cell) and the "
                    "v1.5 permutation test could not separate it from noise "
                    "(p = 0.6064)"}


def cluster_bootstrap(df: pd.DataFrame, n_boot: int = 8000, seed: int = 20261104) -> dict:
    """Participant-level cluster bootstrap: the participant is the resampling unit."""
    rng = np.random.default_rng(seed)
    parts = sorted(set(df["participant"]))
    idx_by = {p: df[df.participant == p] for p in parts}
    out = {k: [] for k in ("E1", "E1b", "E2", "E2b", "E4")}
    n = len(parts)
    for _ in range(n_boot):
        pick = [parts[i] for i in rng.integers(0, n, size=n)]
        sub = pd.concat([idx_by[p] for p in pick], ignore_index=True)
        try:
            e = compute_estimands(sub)
        except Exception:
            continue
        for key, k in (("E1", "E1_pooled_supply_share"),
                       ("E1b", "E1b_subject_equal_supply_share"),
                       ("E2", "E2_within_person_activity_gradient"),
                       ("E2b", "E2b_positive_gradient_fraction"),
                       ("E4", "E4_device_refusal_rate")):
            v = e[k]
            if np.isfinite(v):
                out[key].append(v)
    res = {}
    for k, arr in out.items():
        a = np.array(arr, float)
        if len(a) == 0:
            res[k] = {"ci95": [None, None], "ci95_width": None, "replicates": 0}
            continue
        lo, hi = np.quantile(a, [0.025, 0.975])
        res[k] = {"ci95": [float(lo), float(hi)],
                  "ci95_width": float(hi - lo),
                  "bootstrap_median": float(np.median(a)),
                  "replicates": int(len(a))}
    return res


# ---------------------------------------------------------- the criteria
def exact_signflip_p(deltas) -> float:
    """EXACT one-sided sign-flip permutation p-value on the participant-level
    paired log-odds deltas.

    All 2^n sign patterns are enumerated, so there is no Monte-Carlo error and
    the attainable p-values are multiples of 1/2^n. Under a symmetric null the
    Type I error is controlled by construction.

    Added in v1.7. See evidence/v1_7/v1_7_criterion_audit.json.
    """
    d = np.asarray([v for v in deltas if np.isfinite(v)], float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    cnt = 0
    for s in itertools.product([1, -1], repeat=n):
        if float(np.mean(np.array(s, float) * d)) >= obs - 1e-12:
            cnt += 1
    return cnt / float(2 ** n)


def exact_sign_test_p(deltas) -> float:
    """EXACT one-sided binomial sign test on the number of positive deltas.

    Assumption-light companion to the sign-flip test: the null is
    P(delta > 0) = 1/2 and no symmetry of the magnitudes is required.
    """
    d = np.asarray([v for v in deltas if np.isfinite(v)], float)
    n = len(d)
    if n == 0:
        return float("nan")
    k = int((d > 0).sum())
    if k == 0:
        return 1.0
    return float(sps.binom.sf(k - 1, n, 0.5))


def evaluate_criteria(est: dict, boot: dict) -> dict:
    """Apply the FROZEN confirmation criteria to the computed estimates.

    v1.7 criteria (see evidence/v1_7/v1_7_criterion_audit.json):

    E1  pooled supply share. CONFIRMATORY. The rule is INTERVAL based:
        the whole 95% participant-level bootstrap interval of the pooled
        share lies below 0.50  => SUPPORTED (refusal dominance confirmed);
        the interval contains 0.50  => UNTESTED against the benchmark;
        the interval lies entirely above 0.50 => REFUTED.
        The v1.6 point rule ("point estimate < 0.50") is WITHDRAWN: at a
        true share of exactly 0.50 an unbiased estimator fires about half
        the time, so the point rule carries no confidence statement.

    E2  within-person activity gradient. DESCRIPTIVE (v1.7 demotion).
        No admissible one-sided exact test at n <= 5 meets the pre-registered
        credibility rule: at n = 4 the smallest attainable p is 0.0625 >
        alpha, and at n = 5 the exact test needs all five deltas positive,
        giving a power of about 0.52 against the DEV participant
        distribution. Reported with its exact p-value, never as a
        significance claim.

    E3, E4, E5  DESCRIPTIVE. Reported with values, intervals and
        per-participant detail; no significance claim.
    """
    out = {}
    e1 = est["E1_pooled_supply_share"]
    e1_ci = boot.get("E1", {}).get("ci95", [None, None])
    lo, hi = (e1_ci if isinstance(e1_ci, (list, tuple)) and len(e1_ci) == 2
              else (None, None))
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        status, why = "UNTESTED", "the participant-level bootstrap could not " \
                                 "produce a finite interval"
    elif hi < 0.50:
        status, why = "SUPPORTED", ("the entire 95% interval lies below "
                                    "0.50, so refusal dominance is confirmed "
                                    "with a confidence statement")
    elif lo <= 0.50 <= hi:
        status, why = "UNTESTED", ("the 95% interval contains 0.50, so this "
                                    "cohort is compatible both with refusal "
                                    "dominance and with its absence; the "
                                    "benchmark is UNTESTED, never refuted")
    else:
        status, why = "REFUTED", ("the entire 95% interval lies above 0.50, "
                                   "so refusal does NOT dominate in this "
                                   "cohort")
    out["E1_pooled_supply_share"] = {
        "point_estimate": e1,
        "ci95": [lo, hi],
        "criterion": "the entire 95% participant-level bootstrap interval of "
                     "the pooled supply share lies below 0.50",
        "status": status,
        "reason": why,
        "tier": "PRIMARY / CONFIRMATORY",
    }

    deltas = est["E2_within_person_activity_gradient_raw"]["per_participant"]
    vals = [v for v in deltas.values() if np.isfinite(v)]
    out["E2_within_person_activity_gradient"] = {
        "point_estimate": est["E2_within_person_activity_gradient"],
        "ci95": boot.get("E2", {}).get("ci95", [None, None]),
        "exact_signflip_p_one_sided": exact_signflip_p(vals),
        "exact_sign_test_p_one_sided": exact_sign_test_p(vals),
        "positive_fraction": est["E2b_positive_gradient_fraction"],
        "per_participant": deltas,
        "criterion": "NONE -- DESCRIPTIVE",
        "status": "DESCRIPTIVE",
        "reason": "no admissible one-sided exact test at n <= 5 satisfies the "
                  "pre-registered credibility rule; at n = 4 the smallest "
                  "attainable one-sided p is 0.0625 > alpha = 0.05 and at "
                  "n = 5 the exact test requires all five participant deltas "
                  "to be positive (power about 0.52). A null result is "
                  "UNTESTED, never refuted.",
        "tier": "DESCRIPTIVE ONLY (v1.7 demotion)",
    }
    out["E4_device_refusal_rate"] = {
        "point_estimate": est["E4_device_refusal_rate"],
        "ci95": boot.get("E4", {}).get("ci95", [None, None]),
        "per_participant": est["E4_raw"]["per_participant_refusal_rate"],
        "criterion": "NONE -- DESCRIPTIVE",
        "status": "DESCRIPTIVE",
        "reason": "the v1.6 correction stands: the tested null (rate = 0) is "
                  "trivial, so the quantity is reported as a magnitude with "
                  "an interval and per-participant values, with no "
                  "significance claim",
        "tier": "DESCRIPTIVE ONLY",
    }
    out["E3_between_person_spread"] = {
        "point_estimate": est["E3_between_person_spread"],
        "ci95": None,
        "ci95_note": "the participant-level bootstrap does not resample the "
                     "quartile ratio, so no interval is claimed for E3; the "
                     "per-participant conditional error rates are reported "
                     "instead",
        "per_participant": est["E3_raw"]["per_participant_error_rate"],
        "criterion": "NONE -- DESCRIPTIVE",
        "status": "DESCRIPTIVE",
        "tier": "DESCRIPTIVE ONLY",
    }
    out["E5_eta2_participant_share"] = {
        "point_estimate": est["E5_eta2_participant_share"],
        "criterion": "NONE -- DESCRIPTIVE",
        "status": "DESCRIPTIVE",
        "note": "no eta-squared quantity is reported in the confirmation "
                "side of the paper; the v1.4 interaction effect size was "
                "withdrawn and the cell term is not separable from binomial "
                "sampling error",
        "tier": "DESCRIPTIVE ONLY",
    }
    return out


def paper_label(criteria: dict, evaluable_n: int) -> dict:
    """The pre-registered paper label. Written BEFORE any SEALED value exists.

    With E2 demoted to descriptive (v1.7), the confirmation can support only
    the supply-boundary component; the activity component is reported as an
    observation, not as a tested claim.
    """
    if evaluable_n < 4:
        return {"label": "CONFIRMATION_INCONCLUSIVE",
                "reason": "fewer than four evaluable participants; every "
                          "estimand is reported descriptively"}
    e1 = criteria["E1_pooled_supply_share"]["status"]
    if e1 == "SUPPORTED":
        return {"label": "CONFIRMED_NARROW_CLAIM",
                "reason": "the supply-boundary component is independently "
                          "supported; the activity component is reported as a "
                          "descriptive within-person observation because the "
                          "held-out cohort was not powered to test it"}
    if e1 == "UNTESTED":
        return {"label": "CLAIM_WITH_LIMITATIONS",
                "reason": "the supply-boundary benchmark could not be "
                          "resolved in this cohort; the share is reported as "
                          "cohort-dependent"}
    return {"label": "CLAIM_NOT_SUPPORTED",
            "reason": "the independent result conflicts with the "
                      "supply-boundary component: refusal does not dominate "
                      "in this cohort"}


# --------------------------------------------------------------- the stages
def stage_dev_verify(args) -> dict:
    """Run the IDENTICAL computation on DEV and assert the frozen values."""
    raw = Path(args.raw)
    df = pd.read_csv(HERE / "evidence" / "pivot" / "pivot_features.csv")
    df = df.rename(columns={"bad10": "bad10"})
    df["bad10"] = df["bad10"] == True
    # Rebuild the frozen label columns from the frozen table, then verify the
    # reference values. The window table itself was produced by the same code
    # path (audit_pivot_endpoint.build_universe), so this stage checks the
    # ESTIMAND layer end to end.
    df["ref_ok"] = df["ref_median"].notna()
    df["watch_missing"] = df["watch_missing"] == True
    df["group"] = df["session"].map(
        {s: g for g, m in ACTIVITY_GROUPS.items() for s in m})

    # --- LAYER 1: the ESTIMAND layer on the frozen DEV table -------------
    est = compute_estimands(df)
    checks = []
    pairs = [("pooled_supply_share", est["E1_pooled_supply_share"]),
             ("subject_equal_supply_share", est["E1b_subject_equal_supply_share"]),
             ("within_person_activity_gradient", est["E2_within_person_activity_gradient"]),
             ("error_rate_q75_over_q25", est["E3_between_person_spread"]),
             ("mean_refusal_rate", est["E4_device_refusal_rate"]),
             ("eta2_participant", est["E5_eta2_participant_share"])]
    for name, got in pairs:
        want = DEV_REFERENCE[name]
        ok = abs(got - want) <= DEV_TOLERANCE
        checks.append({"quantity": name, "frozen_reference": want,
                       "recomputed": float(got),
                       "abs_diff": float(abs(got - want)),
                       "tolerance": DEV_TOLERANCE, "pass": bool(ok)})
    e5 = est["E5_detail"]
    for name, key in (("eta2_group", "eta2_group"), ("eta2_cell", "eta2_cell")):
        want = DEV_REFERENCE[name]
        got = e5[key]
        ok = abs(got - want) <= 1e-3
        checks.append({"quantity": name, "frozen_reference": want,
                       "recomputed": float(got),
                       "abs_diff": float(abs(got - want)),
                       "tolerance": 1e-3, "pass": bool(ok)})

    # --- LAYER 2: the WINDOW layer, recomputed from raw DEV files ---------
    # Rebuild the window table for every DEV participant straight from the raw
    # files and compare it to the frozen pivot table row by row. This is what
    # proves that the SEALED run will use the same labelling code.
    audit = json.loads((HERE / "evidence" / "raw_audit_dev.json").read_text(encoding="utf-8"))
    frozen = pd.read_csv(HERE / "evidence" / "pivot" / "pivot_features.csv")
    frozen["ref_ok"] = frozen["ref_median"].notna()
    layer2 = {"participants_rebuilt": [], "mismatches": [], "status": "NOT_RUN"}
    rebuilt_rows = []
    for sid in sorted(set(frozen["participant"])):
        a = audit["subjects"].get(sid, {})
        if a.get("state") != "READY_FOR_WINDOW_CHECK":
            continue
        off = a["time_alignment"]["selected_offset_ms"]
        d, _ = label_participant(raw, sid, off)
        rebuilt_rows.append(d)
        layer2["participants_rebuilt"].append(sid)
    if rebuilt_rows:
        rb = pd.concat(rebuilt_rows, ignore_index=True)
        key = ["participant", "session", "start_ms"]
        m = frozen.merge(rb, on=key, how="outer", suffixes=("_fz", "_rb"),
                         indicator=True)
        both = m[m._merge == "both"]
        for col in ["watch_ok", "bad10", "watch_missing", "ref_ok", "end_ms"]:
            a_ = both[f"{col}_fz"].astype(str).values
            b_ = both[f"{col}_rb"].astype(str).values
            bad = int(np.sum(a_ != b_))
            layer2["mismatches"].append({"column": col, "n_compared": int(len(both)),
                                         "n_differing": bad})
        layer2["status"] = "OK" if all(x["n_differing"] == 0
                                       for x in layer2["mismatches"]) else "MISMATCH"
        layer2["n_rows_compared"] = int(len(both))
        layer2["n_only_in_frozen"] = int((m._merge == "left_only").sum())
        layer2["n_only_in_rebuilt"] = int((m._merge == "right_only").sum())

    all_pass = all(c["pass"] for c in checks) and layer2["status"] == "OK"

    # --- LAYER 3: the v1.7 CRITERION layer, exercised on DEV --------------
    # The confirmation criteria must be executable and must give the expected
    # answer on the DEV cohort, otherwise the SEALED run could not be judged.
    boot_dev = cluster_bootstrap(df, n_boot=8000)
    crit_dev = evaluate_criteria(est, boot_dev)
    label_dev = paper_label(crit_dev, est["n_evaluable"])
    crit_checks = [
        {"quantity": "E1 criterion status on DEV",
         "expected": "SUPPORTED", "got": crit_dev["E1_pooled_supply_share"]["status"],
         "pass": crit_dev["E1_pooled_supply_share"]["status"] == "SUPPORTED"},
        {"quantity": "E2 tier on DEV",
         "expected": "DESCRIPTIVE", "got": crit_dev["E2_within_person_activity_gradient"]["tier"],
         "pass": crit_dev["E2_within_person_activity_gradient"]["tier"].startswith("DESCRIPTIVE")},
        {"quantity": "E4 tier on DEV",
         "expected": "DESCRIPTIVE", "got": crit_dev["E4_device_refusal_rate"]["tier"],
         "pass": crit_dev["E4_device_refusal_rate"]["tier"].startswith("DESCRIPTIVE")},
        {"quantity": "E2 exact sign-flip p is finite and < 1",
         "expected": "finite in (0,1)",
         "got": crit_dev["E2_within_person_activity_gradient"]["exact_signflip_p_one_sided"],
         "pass": bool(np.isfinite(crit_dev["E2_within_person_activity_gradient"]
                                  ["exact_signflip_p_one_sided"]))},
        {"quantity": "E2 exact sign test p is finite and < 1",
         "expected": "finite in (0,1)",
         "got": crit_dev["E2_within_person_activity_gradient"]["exact_sign_test_p_one_sided"],
         "pass": bool(np.isfinite(crit_dev["E2_within_person_activity_gradient"]
                                  ["exact_sign_test_p_one_sided"]))},
        {"quantity": "paper label on DEV",
         "expected": "CONFIRMED_NARROW_CLAIM", "got": label_dev["label"],
         "pass": label_dev["label"] == "CONFIRMED_NARROW_CLAIM"},
    ]
    crit_pass = all(c["pass"] for c in crit_checks)

    doc = {
        "stage": "dev-verify",
        "state": "DEV_VERIFY_OK" if (all_pass and crit_pass) else "DEV_VERIFY_FAILED",
        "run_type": "MEASUREMENT ONLY -- zero fitting, zero threshold search",
        "gate": "G0.4: the same code path used for the SEALED confirmation must "
                "reproduce the frozen DEV reference values bit-for-bit",
        "layer1_estimand_layer": {"checks": checks,
                                  "all_pass": all(c["pass"] for c in checks)},
        "layer2_window_layer": layer2,
        "layer3_criterion_layer": {
            "checks": crit_checks,
            "all_pass": crit_pass,
            "note": "the v1.7 criteria are exercised on DEV so that the "
                    "SEALED run can be judged; E2 is demoted to descriptive "
                    "and E1 uses the interval rule",
        },
        "dev_criteria": crit_dev,
        "dev_paper_label": label_dev,
        "estimates": {k: v for k, v in est.items() if k.startswith("E")},
        "per_participant": est["per_participant"],
        "sealed_UNREAD": OUR_SEALED,
        "no_sealed_file_opened": True,
    }
    (OUT / "v1_6_dev_verify.json").write_text(
        json.dumps(json_safe(doc), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "state": doc["state"],
        "layer1_all_pass": doc["layer1_estimand_layer"]["all_pass"],
        "layer2_status": layer2["status"],
        "layer2_mismatches": layer2["mismatches"],
        "layer3_all_pass": crit_pass,
        "layer3_failures": [c for c in crit_checks if not c["pass"]],
        "dev_paper_label": label_dev["label"],
        "out": str(OUT / "v1_6_dev_verify.json"),
    }, indent=2, ensure_ascii=False))
    if not (all_pass and crit_pass):
        sys.exit(1)
    return doc


def _require_authorisation(stage: str) -> None:
    if os.environ.get("HRGUARD_SEALED_AUTHORIZED") != "1":
        raise PermissionError(
            f"stage '{stage}' requires explicit user authorisation for the "
            "SEALED unseal. Set HRGUARD_SEALED_AUTHORIZED=1 only after that "
            "authorisation has been given.")


def stage_preflight(args) -> dict:
    """Frozen clock rule over the SEALED participants. No labels, no estimands."""
    _require_authorisation("preflight")
    raw = Path(args.raw)
    diag = {}
    for sid in OUR_SEALED:
        missing = [f for f in REQUIRED_FILES if not (raw / sid / f).exists()]
        if missing:
            diag[sid] = {"state": "DATA_NOT_EVALUABLE",
                         "reason": "missing required raw file(s): " + ", ".join(missing),
                         "missing": missing,
                         "rule": "G-EVAL clause (1): all five raw files present",
                         "note": "a missing file makes 'the device emitted "
                                 "nothing' indistinguishable from 'the file was "
                                 "not recorded'; it is NOT counted as a refusal "
                                 "and enters neither a numerator nor a denominator"}
            continue
        src_hr = read_minimal(raw / sid / "GalaxyWatch/HR.csv", ("timestamp", "hr", "hrStatus"))
        src_ppg = read_minimal(raw / sid / "GalaxyWatch/PPG.csv", ("timestamp",))
        polar = read_minimal(raw / sid / "PolarH10/HR.csv", ("phoneTimestamp", "hr"))
        p_t = pd.to_numeric(polar["phoneTimestamp"], errors="coerce").to_numpy(float)
        p_t = p_t[np.isfinite(p_t)]
        w_ppg_t = pd.to_numeric(src_ppg["timestamp"], errors="coerce").to_numpy(float)
        w_ppg_t = w_ppg_t[np.isfinite(w_ppg_t)]
        w_hr_t = pd.to_numeric(src_hr["timestamp"], errors="coerce").to_numpy(float)
        w_hr_t = w_hr_t[np.isfinite(w_hr_t)]
        # V1.7 FIX (defect found at the first SEALED preflight, 2026-11-04):
        # infer_offset(reference_times, watch_times) adds the candidate offset to
        # the REFERENCE (Polar) timestamps and measures how many of them land
        # inside the WATCH range. The earlier call passed the arguments in the
        # opposite order, which makes both candidate scores identically 0.0 and
        # turns every participant into DATA_NOT_EVALUABLE on a bookkeeping
        # technicality. The frozen semantics are those of
        # audit_raw_dev.infer_offset(polar, watch_ppg); DEV reference behaviour is
        # reproduced by test_clock_rule.py (0.0 / 0.97 for every DEV subject).
        dec = infer_offset(p_t, w_ppg_t)
        guard = clock_guard(p_t.min(), p_t.max(), w_hr_t.min(), w_hr_t.max(),
                            dec["selected_offset_ms"])
        ok = dec["selected_offset_ms"] is not None and guard["passed"]
        diag[sid] = {
            "state": "ALIGNMENT_PASS" if ok else "DATA_NOT_EVALUABLE",
            "rule_applied": "timestamp overlap only; reference-HR error never used",
            "decision": dec, "guard": guard,
            "reason": "" if ok else ("clock rule failed" if dec["selected_offset_ms"] is None
                                    else "continuous residual envelope violated"),
        }
    doc = {
        "stage": "preflight",
        "state": "SEALED_PREFLIGHT_WRITTEN",
        "written_before_any_label": True,
        "rule": "closed candidate set {raw:0, author_minus_9h:-32400000}; "
                "score = fraction of Polar timestamps inside the watch PPG "
                "timestamp range; PASS iff best >= 0.75 AND (best-second) >= "
                "0.30 AND res_lo_s in [-578.8,+109.1] AND res_hi_s in "
                "[-509.4,+130.9]",
        "prohibition": "no reference-HR error may be used to choose or justify "
                       "an offset, and no sibling-project HR-derived quantity "
                       "may be used to accept or reject a participant",
        "per_participant": diag,
        "evaluable": [s for s, v in diag.items() if v["state"] == "ALIGNMENT_PASS"],
        "not_evaluable": {s: v["reason"] for s, v in diag.items()
                          if v["state"] != "ALIGNMENT_PASS"},
    }
    (OUT / "v1_6_sealed_preflight.json").write_text(
        json.dumps(json_safe(doc), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evaluable": doc["evaluable"],
                      "not_evaluable": doc["not_evaluable"],
                      "out": str(OUT / "v1_6_sealed_preflight.json")},
                     indent=2, ensure_ascii=False))
    return doc


def stage_confirm(args) -> dict:
    """The one-shot measurement run. Zero fitting."""
    _require_authorisation("confirm")
    raw = Path(args.raw)
    ledger = OUT / "v1_6_confirmation_ledger.json"
    if ledger.exists():
        raise RuntimeError("A confirmation run already exists. The run is "
                           "single-shot and cannot be repeated.")
    pre = json.loads((OUT / "v1_6_sealed_preflight.json").read_text(encoding="utf-8"))

    frames = []
    excluded = {}
    for sid in OUR_SEALED:
        st = pre["per_participant"].get(sid, {})
        if st.get("state") != "ALIGNMENT_PASS":
            excluded[sid] = {"state": "DATA_NOT_EVALUABLE",
                             "reason": st.get("reason", "unknown"),
                             "enters_numerator": False, "enters_denominator": False}
            continue
        off = st["decision"]["selected_offset_ms"]
        d, _ = label_participant(raw, sid, off)
        frames.append(d)

    if not frames:
        doc = {"stage": "confirm", "state": "NO_EVALUABLE_PARTICIPANT",
               "excluded": excluded,
               "note": "fewer than the required 4 evaluable participants; "
                       "every estimand is downgraded to descriptive reporting"}
    else:
        df = pd.concat(frames, ignore_index=True)
        est = compute_estimands(df)
        boot = cluster_bootstrap(df, n_boot=args.n_boot)
        crit = evaluate_criteria(est, boot)
        label = paper_label(crit, est["n_evaluable"])
        doc = {
            "stage": "confirm",
            "state": "SEALED_CONFIRMATION_COMPLETED",
            "run_type": "MEASUREMENT ONLY -- zero fitting, zero threshold "
                        "calibration, zero hyper-parameter search",
            "one_shot": True,
            "evaluable_n": est["n_evaluable"],
            "excluded": excluded,
            "estimands": est,
            "participant_cluster_bootstrap": boot,
            "criteria": crit,
            "paper_label": label,
            "n_boot": args.n_boot,
            "seed": 20261104,
        }
    ledger.write_text(json.dumps(json_safe(doc), indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps(json_safe({
        "state": doc["state"], "evaluable_n": doc.get("evaluable_n"),
        "excluded": list(excluded.keys()),
        "paper_label": doc.get("paper_label"),
        "out": str(ledger)}), indent=2, ensure_ascii=False))
    return doc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["dev-verify", "preflight", "confirm"])
    ap.add_argument("--raw", default=str(HERE / "raw_dev_only"))
    ap.add_argument("--n-boot", type=int, default=8000)
    args = ap.parse_args()
    if args.stage == "dev-verify":
        stage_dev_verify(args)
    elif args.stage == "preflight":
        stage_preflight(args)
    else:
        stage_confirm(args)


if __name__ == "__main__":
    main()
