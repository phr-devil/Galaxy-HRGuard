"""Pivot feature table: reuse the v0.5 DEV features, extract only the 2,747 missing windows.

The old feature file dropped every window in which the watch reported no usable HR, which is
precisely the population the PIVOT endpoint needs. This script reuses the 22,177 existing rows
verbatim and re-derives features from raw signals only for the uncovered windows, using the
same frozen 8 s / 2 s grid, the same window geometry and the same spectral code.

No fitting, no sealed access. Features are history-window (H) only by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .raw import FILES, read_minimal
from .signals import uniform_signal
from .core import DEV, SEAL, norm_spectrum, sha256_file
from .windows import (MIN_GOOD_HR, clean_series, sensor_is_usable,
                               sessions_from_events, window_slice)

HR_FEATS = ["watch_hr", "valid_hr_fraction", "watch_hr_last_age_ms",
            "vendor_hr_std", "vendor_hr_delta"]
SIG_FEATS = ["ppg_concentration", "ppg_entropy", "ppg_peak_exists",
             "ppg_hr_disagree", "acc_motion", "acc_ppg_overlap"]
ALL_FEATS = HR_FEATS + ["polar_hr_last_age_ms"] + SIG_FEATS


def signal_features_optional_hr(ppg, acc, watch_hr):
    """Same legal features as hrguard.signal_features, but watch HR may be absent.

    When the vendor algorithm refused to output HR in H, that absence is itself the
    strongest legal input, so the row is kept and only the HR-derived disagreement
    becomes NaN instead of deleting the window.
    """
    if len(ppg) != 200 or len(acc) != 200:
        return None
    ps = norm_spectrum(ppg.flatten(), 25)
    if ps is None:
        ppg_peak_hr, concentrated, ent, peak_exists = np.nan, 0.0, 1.0, 0.0
    else:
        f, p = ps
        peak_idx = int(np.argmax(p))
        ppg_peak_hr = 60.0 * f[peak_idx]
        concentrated = float(np.max(p))
        ent = float(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(len(p)))
        peak_exists = 1.0
    dyn = acc - acc.mean(axis=0, keepdims=True)
    motion = float(np.sqrt(np.mean(np.sum(dyn ** 2, axis=1))))
    spectra = []
    for j in range(3):
        q = norm_spectrum(dyn[:, j], 25)
        if q is not None and ps is not None:
            spectra.append(q[1])
    overlap = float(np.max(np.minimum(p, np.mean(spectra, axis=0)))) if spectra else 0.0
    disagree = np.nan
    if np.isfinite(watch_hr):
        disagree = float(abs(ppg_peak_hr - watch_hr)) if peak_exists else 200.0
    return {"ppg_concentration": concentrated, "ppg_entropy": ent,
            "ppg_peak_exists": peak_exists, "ppg_hr_disagree": disagree,
            "acc_motion": motion, "acc_ppg_overlap": overlap}


def hr_window_features(wh, t0, t1):
    """Vendor HR state inside H. Absence of valid HR is an input, not a reason to drop."""
    h = window_slice(wh, t0, t1)
    hs = pd.to_numeric(h.hrStatus, errors="coerce")
    hr = pd.to_numeric(h.hr, errors="coerce")
    good = h.loc[(hs == 1) & hr.between(40, 200)]
    hv = pd.to_numeric(good.hr, errors="coerce").to_numpy(dtype=float)
    out = {"watch_hr": np.nan, "valid_hr_fraction": float(len(hv)) / 8.0,
           "watch_hr_last_age_ms": np.nan, "vendor_hr_std": np.nan,
           "vendor_hr_delta": np.nan}
    if len(hv) >= MIN_GOOD_HR:
        out.update({"watch_hr": float(np.median(hv)),
                    "valid_hr_fraction": min(1.0, len(hv) / 8.0),
                    "watch_hr_last_age_ms": float(t1 - good.timestamp.iloc[-1]),
                    "vendor_hr_std": float(np.std(hv)),
                    "vendor_hr_delta": float(abs(hv[-1] - hv[0]))})
    return out, len(hv) >= MIN_GOOD_HR


def build(raw: Path, audit_json: Path, universe_csv: Path, old_features_csv: Path,
          out_csv: Path, out_receipt: Path):
    facts = json.loads(audit_json.read_text(encoding="utf-8"))
    if facts.get("state") != "READ_ONLY_DEV_AUDIT_NO_FIT":
        raise ValueError("Need validated raw audit")
    uni = pd.read_csv(universe_csv)
    old = pd.read_csv(old_features_csv)
    if not set(uni.participant.unique()).issubset(DEV):
        raise PermissionError("Universe references non-DEV participants")
    key = ["participant", "session", "start_ms"]
    have = set(map(tuple, old[key].to_numpy()))
    todo = uni.loc[~uni.apply(lambda r: (r.participant, r.session, r.start_ms) in have, axis=1)]
    todo = todo.sort_values(key)
    print(f"reusing {len(old)} existing rows; extracting {len(todo)} uncovered windows")

    new_rows = []
    for sid, g in todo.groupby("participant"):
        if sid not in DEV:
            raise PermissionError(f"Sealed or unknown subject {sid}")
        a = facts["subjects"][sid]
        if a.get("state") != "READY_FOR_WINDOW_CHECK":
            raise ValueError(f"{sid} not audited")
        sources = {}
        for name, (rel, cols) in FILES.items():
            src = raw / sid / rel
            if sha256_file(src) != a["sha256"].get(name):
                raise ValueError(f"Raw file changed after audit: {sid} {name}")
            sources[name] = read_minimal(src, cols)
        offset = a["time_alignment"]["selected_offset_ms"]
        ppg = clean_series(sources["watch_ppg"], "timestamp", ["ppg", "status"])
        acc = clean_series(sources["watch_acc"], "timestamp", ["x", "y", "z"])
        wh = clean_series(sources["watch_hr"], "timestamp", ["hr", "hrStatus"])
        ref = clean_series(sources["polar_hr"].rename(columns={"phoneTimestamp": "timestamp"}),
                           "timestamp", ["hr"])
        ref["timestamp"] += offset
        want = {(r.session, int(r.start_ms)) for r in g.itertuples()}
        for session, s0, s1 in sessions_from_events(sources["events"]):
            for t0 in range(s0, s1 - 8000 + 1, 2000):
                if (session, t0) not in want:
                    continue
                t1 = t0 + 8000
                r = window_slice(ref, t0, t1)
                rv = pd.to_numeric(r.hr, errors="coerce").to_numpy(dtype=float)
                rv = rv[np.isfinite(rv) & (rv >= 40) & (rv <= 200)]
                polar_age = float(t1 - r.loc[np.isfinite(pd.to_numeric(r.hr, errors="coerce"))
                                             & pd.to_numeric(r.hr, errors="coerce").between(40, 200)]
                                  .timestamp.iloc[-1]) if len(rv) else np.nan
                hrf, ok = hr_window_features(wh, t0, t1)
                unused_ok = ok
                p_ok = sensor_is_usable(ppg, t0, t1, good_status=0)
                a_ok = sensor_is_usable(acc, t0, t1)
                x = uniform_signal(ppg, ["ppg"], t0, t1, status=0) if p_ok else None
                z = uniform_signal(acc, ["x", "y", "z"], t0, t1) if a_ok else None
                if x is not None and z is not None:
                    feat = signal_features_optional_hr(x, z, hrf["watch_hr"])
                    if feat is None:
                        raise ValueError("Feature failure after quality gate")
                else:
                    feat = {k: np.nan for k in SIG_FEATS}
                new_rows.append({"participant": sid, "session": session, "start_ms": t0,
                                 "polar_hr_last_age_ms": polar_age, **hrf, **feat})

    new = pd.DataFrame(new_rows)
    merged = pd.concat([old, new], ignore_index=True, sort=False)
    merged = uni[key + ["end_ms", "ref_median", "n_watch_any", "n_watch_valid", "watch_ok",
                        "ppg_ok", "acc_ok", "y_unreliable", "y_missing", "y_error",
                        "bad10", "watch_missing"]].merge(
        merged[key + HR_FEATS + ["polar_hr_last_age_ms"] + SIG_FEATS], on=key, how="left")
    if len(merged) != len(uni) or merged[key].duplicated().any():
        raise ValueError("Merged table does not match the frozen universe")
    if merged[HR_FEATS[1]].isna().any() and False:
        raise ValueError("Missing validity fraction")
    for c in ["valid_hr_fraction"]:
        if merged[c].isna().any():
            raise ValueError(f"NaN in {c}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    h = hashlib.sha256()
    with out_csv.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    receipt = {"state": "PIVOT_FEATURES_EXTRACTED_NO_FIT", "protocol": "Galaxy-HRGuard-v0.9-pivot",
               "universe_csv": str(universe_csv), "universe_sha256": sha256_file(universe_csv),
               "reused_old_features": str(old_features_csv),
               "reused_rows": int(len(old)), "newly_extracted_rows": int(len(new)),
               "pivot_features_sha256": h.hexdigest(), "N": int(len(merged)),
               "participants": int(merged.participant.nunique()),
               "y_unreliable": int(merged.y_unreliable.sum()),
               "y_missing": int(merged.y_missing.sum()), "y_error": int(merged.y_error.sum()),
               "feature_columns": HR_FEATS + SIG_FEATS,
               "sealed_UNREAD": sorted(SEAL)}
    out_receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--universe", required=True, type=Path)
    ap.add_argument("--old-features", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--receipt", required=True, type=Path)
    a = ap.parse_args()
    build(a.raw, a.audit, a.universe, a.old_features, a.out, a.receipt)


if __name__ == "__main__":
    main()
