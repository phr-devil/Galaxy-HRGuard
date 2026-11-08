"""Read-only audit of the PIVOT endpoint: future 8-10 s watch-HR unreliability (missing OR gross error).

No fitting, no sealed access, no model. Produces the per-window universe (including
windows where the watch reported no HR, which the old pipeline dropped) and the
learnability evidence needed to decide whether the new task is pilotable.

Prediction geometry (frozen here, chosen for causality not performance):
    history H = [s, s+8000)      inputs only from H and earlier
    target  T = [s+10000, s+18000)   future window, 2 s lead, no overlap with H
    Y(T)    = 1 if the watch reports no valid HR in T, or |median(watch HR) - mean(Polar HR)| > 10 bpm
History inputs never touch T, so the current missing state of T cannot leak into inputs.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .raw import FILES, read_minimal
from .core import DEV, SEAL, sha256_file
from .windows import (WINDOW_MS, STRIDE_MS, MIN_GOOD_HR, clean_series,
                               sensor_is_usable, sessions_from_events, window_slice)

TARGET_LEAD_MS = 10000  # T starts 10 s after H starts -> 2 s gap after H ends


def build_universe(raw: Path, audit_json: Path, out_csv: Path):
    facts = json.loads(audit_json.read_text(encoding="utf-8"))
    if facts.get("state") != "READ_ONLY_DEV_AUDIT_NO_FIT":
        raise ValueError("Need validated raw audit")
    rows = []
    for sid in sorted(DEV):
        a = facts.get("subjects", {}).get(sid, {})
        if a.get("state") != "READY_FOR_WINDOW_CHECK":
            continue
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
                p_ok = sensor_is_usable(ppg, t0, t1, good_status=0)
                a_ok = sensor_is_usable(acc, t0, t1)   # evaluated even when PPG fails
                ref_ok = len(rv) >= MIN_GOOD_HR
                watch_ok = len(hv) >= MIN_GOOD_HR
                err = float(abs(np.median(hv) - np.mean(rv))) if (ref_ok and watch_ok) else np.nan
                rows.append({"participant": sid, "session": session, "start_ms": t0, "end_ms": t1,
                             "n_ref": int(len(rv)), "ref_median": float(np.mean(rv)) if ref_ok else np.nan,
                             "ref_ok": bool(ref_ok),
                             "n_watch_any": int(len(h)), "n_watch_valid": int(len(hv)),
                             "watch_ok": bool(watch_ok),
                             "watch_median": float(np.median(hv)) if watch_ok else np.nan,
                             "ppg_ok": bool(p_ok), "acc_ok": bool(a_ok),
                             "error_bpm": err,
                             "bad10": bool(np.isfinite(err) and err > 10)})
    df = pd.DataFrame(rows)
    df["watch_missing"] = df.ref_ok & ~df.watch_ok
    df["labelable"] = df.ref_ok
    # Endpoint: among reference-valid windows, unreliable = no reported HR or gross error.
    df["y_unreliable"] = np.where(df.labelable, (df.watch_missing | df.bad10).astype(float), np.nan)
    df["y_missing"] = np.where(df.labelable, df.watch_missing.astype(float), np.nan)
    df["y_error"] = np.where(df.labelable, df.bad10.astype(float), np.nan)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df, facts


def episodes(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Contiguous runs of the flag within participant+session (2 s stride, <=2.5 s gap)."""
    out = []
    for (pid, sess), g in df.groupby(["participant", "session"], sort=False):
        g = g.sort_values("start_ms")
        v = g[flag].to_numpy(dtype=bool)
        st = g.start_ms.to_numpy(dtype=float)
        i = 0
        while i < len(v):
            if v[i]:
                j = i + 1
                while j < len(v) and v[j] and (st[j] - st[j - 1]) <= 2500:
                    j += 1
                n = j - i
                out.append({"participant": pid, "session": sess, "n_windows": n,
                            "duration_ms": float(st[j - 1] - st[i] + WINDOW_MS),
                            "start_ms": float(st[i])})
                i = j
            else:
                i += 1
    return pd.DataFrame(out)


def pair_stats(df: pd.DataFrame):
    """History H -> future target T pairs; descriptive only, no fitting."""
    idx = {(r.participant, r.session, int(r.start_ms)): i
           for i, r in enumerate(df.itertuples())}
    rows = []
    for i, r in enumerate(df.itertuples()):
        j = idx.get((r.participant, r.session, int(r.start_ms) + TARGET_LEAD_MS))
        if j is None:
            continue
        t = df.iloc[j]
        rows.append({"participant": r.participant, "session": r.session,
                     "h_start": r.start_ms, "t_start": t.start_ms,
                     "h_watch_ok": bool(r.watch_ok), "h_watch_missing": bool(r.watch_missing),
                     "h_bad10": bool(r.bad10), "h_ppg_ok": bool(r.ppg_ok), "h_acc_ok": bool(r.acc_ok),
                     "h_valid_frac": float(r.n_watch_valid) / max(int(r.n_watch_any), 1),
                     "t_watch_missing": bool(t.watch_missing), "t_bad10": bool(t.bad10),
                     "t_y": float(t.y_unreliable) if np.isfinite(t.y_unreliable) else np.nan,
                     "t_ref_ok": bool(t.ref_ok)})
    p = pd.DataFrame(rows)
    p = p[p.t_ref_ok]
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out_csv = a.out.with_name(a.out.stem + "_windows.csv")
    df, _ = build_universe(a.raw, a.audit, out_csv)

    lab = df[df.labelable]
    n_lab = len(lab)
    n_miss = int(lab.watch_missing.sum())
    n_bad = int(lab.bad10.sum())
    n_both = int((lab.watch_missing & lab.bad10).sum())
    n_unrel = int((lab.watch_missing | lab.bad10).sum())
    # windows with HR reported but waveform insufficient: old pipeline could not score them
    n_hr_no_wave = int((lab.watch_ok & ~(lab.ppg_ok & lab.acc_ok)).sum())

    ep_u = episodes(df.assign(_f=df.y_unreliable.fillna(False).astype(bool)), "_f")
    ep_m = episodes(df.assign(_f=df.watch_missing.fillna(False).astype(bool)), "_f")
    ep_e = episodes(df.assign(_f=df.bad10.fillna(False).astype(bool)), "_f")

    by_sub = []
    for pid, g in lab.groupby("participant"):
        by_sub.append({"participant": pid, "windows": int(len(g)),
                       "missing": int(g.watch_missing.sum()), "error": int(g.bad10.sum()),
                       "unreliable": int((g.watch_missing | g.bad10).sum()),
                       "unreliable_rate": float((g.watch_missing | g.bad10).mean()),
                       "sessions": int(g.session.nunique())})
    by_sub = sorted(by_sub, key=lambda r: -r["unreliable_rate"])

    by_act = []
    for sess, g in lab.groupby("session"):
        by_act.append({"activity": sess, "windows": int(len(g)),
                       "missing_rate": float(g.watch_missing.mean()),
                       "error_rate_among_reported": float(g.bad10.sum() / max(int(g.watch_ok.sum()), 1)),
                       "unreliable_rate": float((g.watch_missing | g.bad10).mean())})
    by_act = sorted(by_act, key=lambda r: -r["unreliable_rate"])

    # Is ACC still available when PPG is not? (drives which inputs survive at prediction time)
    cross = {"ppg_ok_and_acc_ok": int((df.ppg_ok & df.acc_ok).sum()),
             "ppg_ok_acc_missing": int((df.ppg_ok & ~df.acc_ok).sum()),
             "ppg_missing_acc_ok": int((~df.ppg_ok & df.acc_ok).sum()),
             "both_missing": int((~df.ppg_ok & ~df.acc_ok).sum()),
             "total": int(len(df))}
    cross["acc_available_given_ppg_missing"] = cross["ppg_missing_acc_ok"] / max(
        cross["ppg_missing_acc_ok"] + cross["both_missing"], 1)
    miss_rows = df[df.labelable & df.watch_missing]
    cross["when_watch_hr_missing"] = {
        "n": int(len(miss_rows)),
        "ppg_ok": int(miss_rows.ppg_ok.sum()), "acc_ok": int(miss_rows.acc_ok.sum()),
        "either_ok": int((miss_rows.ppg_ok | miss_rows.acc_ok).sum()),
        "neither_ok": int((~miss_rows.ppg_ok & ~miss_rows.acc_ok).sum())}

    p = pair_stats(df)
    base = float(p.t_y.mean())
    pers = {}
    for name, col in (("H_missing", "h_watch_missing"), ("H_bad10", "h_bad10"),
                      ("H_unreliable", None)):
        if col is None:
            m = (p.h_watch_missing | p.h_bad10)
        else:
            m = p[col].astype(bool)
        pers[name] = {"exposed": int(m.sum()),
                      "p_target_unreliable_when_exposed": float(p.t_y[m].mean()) if m.any() else None,
                      "p_target_unreliable_when_not": float(p.t_y[~m].mean()) if (~m).any() else None,
                      "lift": float(p.t_y[m].mean() / base) if m.any() and base > 0 else None}
    hist_avail = {"n_pairs": int(len(p)),
                  "h_acc_ok": float(p.h_acc_ok.mean()), "h_ppg_ok": float(p.h_ppg_ok.mean()),
                  "h_acc_ok_when_hr_missing_in_H": float(p.h_acc_ok[p.h_watch_missing].mean())
                  if p.h_watch_missing.any() else None,
                  "target_unreliable_rate": base,
                  "target_missing_rate": float(p.t_watch_missing.mean()),
                  "target_error_rate": float(p.t_bad10.mean()),
                  "participants": int(p.participant.nunique())}

    report = {
        "state": "DEV_PIVOT_ENDPOINT_AUDIT_NO_FIT",
        "endpoint": {"name": "future_8_10s_watch_hr_unreliable",
                     "definition": "within T=[s+10s, s+18s] the watch reports no valid HR (>=4 valid, hrStatus==1, 40-200 bpm) OR |median(watch HR)-mean(Polar HR)|>10 bpm",
                     "history": "H=[s, s+8s); inputs strictly from H and earlier",
                     "reference_requirement": "Polar H10 valid in T (>=4 samples, 40-200 bpm)"},
        "universe": {"windows_total": int(len(df)), "reference_valid": n_lab,
                     "watch_hr_missing": n_miss, "gross_error": n_bad,
                     "missing_and_error": n_both, "unreliable_union": n_unrel,
                     "unreliable_rate": n_unrel / n_lab,
                     "hr_reported_but_waveform_insufficient": n_hr_no_wave},
        "count_reconciliation": {
            "missing_plus_error": n_miss + n_bad,
            "note_3080": "3,080 came from applying the raw-universe unreliability fraction to the "
                         "eligible-window denominator (22,177 x 13.93% = 3,089); the window-level "
                         "union on a single denominator is reported as unreliable_union.",
            "note_3449": "3,449 mixed two denominators: missing counted on all reference-valid "
                         "windows, errors counted only inside the waveform-eligible subset.",
            "union_is_not_sum_when_overlap": n_both},
        "episodes": {
            "unreliable": {"n": int(len(ep_u)),
                           "median_duration_ms": float(ep_u.duration_ms.median()) if len(ep_u) else None,
                           "mean_windows": float(ep_u.n_windows.mean()) if len(ep_u) else None,
                           "p90_duration_ms": float(ep_u.duration_ms.quantile(0.9)) if len(ep_u) else None},
            "missing": {"n": int(len(ep_m)),
                        "median_duration_ms": float(ep_m.duration_ms.median()) if len(ep_m) else None},
            "error": {"n": int(len(ep_e)),
                      "median_duration_ms": float(ep_e.duration_ms.median()) if len(ep_e) else None}},
        "by_participant": by_sub,
        "by_activity": by_act,
        "sensor_cross_availability": cross,
        "history_to_future_pairs": hist_avail,
        "persistence_descriptive": pers,
        "sealed_UNREAD": sorted(SEAL),
        "windows_csv": str(out_csv),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"universe": report["universe"], "episodes": report["episodes"],
                      "pairs": hist_avail, "out": str(a.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
