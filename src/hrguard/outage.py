"""RWF v0.9 final review - DEV-only, read-only decisive mechanism test.

Question: when the watch reports PPG and ACC normally but refuses to output HR
(hrStatus != 1), is the optical signal actually destroyed, or does the watch's own
PPG still carry a pulse at the reference (Polar H10) rate?

This separates two competing explanations for the project's central novelty claim
("the loss is algorithmic refusal rather than sensing failure"):

  H_signal : during outage the PPG is corrupted (no recoverable pulse), so refusing
             to report is justified by the input.
  H_refusal: during outage the PPG still contains a pulse at the reference rate, so
             the refusal happens on a signal that was, by this measure, usable.

Design (no fit, DEV only, SEALED never opened):
  * unit = one watch-HR row (~1 Hz), classified ok (hrStatus == 1) vs outage (!= 1)
  * PPG window = the 4 s ENDING at that row (causal), ACC window the same
  * reference rate = Polar H10 HR nearest within +/-5 s (offset -32400000 ms, frozen)
  * confounding control: within (participant, session), seconds are binned by ACC
    motion quintile, and ok vs outage are compared INSIDE each bin; the pooled
    within-session contrast is reported next to it so the size of the activity
    confound stays visible.

Outputs a JSON + prints the two numbers that decide H_signal vs H_refusal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .raw import FILES, read_minimal
from .core import DEV
from .windows import sessions_from_events
from . import paths as _paths

ROOT = _paths.REPO_ROOT
RAW = _paths.data_root()
OUT_DIR = _paths.output_dir("evidence/final_review")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLAR_OFFSET_MS = -32_400_000     # frozen in audit_raw_dev; chosen from timestamps only
PPG_WIN_MS = 4_000
MIN_PPG_SAMPLES = 80              # of the nominal 100 in a 4 s window at 25 Hz
BAND = (0.67, 3.33)               # 40-200 bpm
REF_TOL_HZ = 0.25                 # +/-15 bpm around the reference rate
MATCH_TOL_MS = 5_000
N_ACC_BINS = 5


def band_spectrum(x: np.ndarray, fs: float = 25.0, nfft: int = 512):
    """Hann-windowed periodogram restricted to the HR band; returns (freq, norm power)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < MIN_PPG_SAMPLES:
        return None
    x = x - x.mean()
    w = np.hanning(x.size)
    if np.std(x * w) < 1e-9:
        return None
    spec = np.abs(np.fft.rfft(x * w, n=nfft)) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    ok = (f >= BAND[0]) & (f <= BAND[1])
    if not ok.any():
        return None
    p = spec[ok]
    tot = p.sum()
    if tot <= 1e-15:
        return None
    return f[ok], p / tot


def acc_motion(a: np.ndarray) -> float:
    dyn = a - a.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(dyn ** 2, axis=1))))


def per_second_rows(sid: str) -> pd.DataFrame:
    d = RAW / sid
    ppg = read_minimal(d / FILES["watch_ppg"][0], FILES["watch_ppg"][1])
    acc = read_minimal(d / FILES["watch_acc"][0], FILES["watch_acc"][1])
    whr = read_minimal(d / FILES["watch_hr"][0], FILES["watch_hr"][1])
    pol = read_minimal(d / FILES["polar_hr"][0], FILES["polar_hr"][1])
    ev = read_minimal(d / FILES["events"][0], FILES["events"][1])

    for c in ("timestamp", "ppg", "status"):
        ppg[c] = pd.to_numeric(ppg[c], errors="coerce")
    for c in ("timestamp", "x", "y", "z"):
        acc[c] = pd.to_numeric(acc[c], errors="coerce")
    whr["timestamp"] = pd.to_numeric(whr["timestamp"], errors="coerce")
    whr["hrStatus"] = pd.to_numeric(whr["hrStatus"], errors="coerce")
    whr["hr"] = pd.to_numeric(whr["hr"], errors="coerce")
    pol["phoneTimestamp"] = pd.to_numeric(pol["phoneTimestamp"], errors="coerce")
    pol["hr"] = pd.to_numeric(pol["hr"], errors="coerce")

    ppg = ppg[ppg["status"].fillna(-999) != -999].sort_values("timestamp")
    acc = acc.sort_values("timestamp")
    whr = whr.sort_values("timestamp")
    pol = pol.dropna(subset=["phoneTimestamp", "hr"]).sort_values("phoneTimestamp")
    pol_ts = pol["phoneTimestamp"].to_numpy(float) + POLAR_OFFSET_MS
    pol_hr = pol["hr"].to_numpy(float)

    ppg_t = ppg["timestamp"].to_numpy(float)
    ppg_v = ppg["ppg"].to_numpy(float)
    acc_t = acc["timestamp"].to_numpy(float)
    acc_v = acc[["x", "y", "z"]].to_numpy(float)

    sessions = sessions_from_events(ev)
    sess_of = np.full(len(whr), "", dtype=object)
    w_t = whr["timestamp"].to_numpy(float)
    for name, s0, s1 in sessions:
        m = (w_t >= s0) & (w_t <= s1)
        sess_of[m] = name

    rows = []
    for i, t in enumerate(w_t):
        if not np.isfinite(t) or sess_of[i] == "":
            continue
        t0, t1 = t - PPG_WIN_MS, t
        pi0, pi1 = np.searchsorted(ppg_t, t0, "left"), np.searchsorted(ppg_t, t1, "left")
        seg = ppg_v[pi0:pi1]
        ai0, ai1 = np.searchsorted(acc_t, t0, "left"), np.searchsorted(acc_t, t1, "left")
        aseg = acc_v[ai0:ai1]
        if aseg.size < MIN_PPG_SAMPLES:
            continue
        motion = acc_motion(aseg)
        sp = band_spectrum(seg)
        if sp is None:
            continue
        f, p = sp
        # reference rate
        j = np.searchsorted(pol_ts, t)
        cand = [k for k in (j - 1, j, j + 1) if 0 <= k < len(pol_ts)]
        cand = [k for k in cand if abs(pol_ts[k] - t) <= MATCH_TOL_MS]
        if not cand:
            continue
        k = min(cand, key=lambda q: abs(pol_ts[q] - t))
        ref = float(pol_hr[k])
        if not np.isfinite(ref) or ref <= 0:
            continue
        f0 = ref / 60.0
        conc_ref = float(p[(f >= f0 - REF_TOL_HZ) & (f <= f0 + REF_TOL_HZ)].sum())
        peak_hr = float(60.0 * f[int(np.argmax(p))])
        rows.append({
            "participant": sid,
            "session": sess_of[i],
            "t": float(t),
            "ok": bool(whr["hrStatus"].to_numpy()[i] == 1),
            "acc_motion": motion,
            "ppg_n": int(seg.size),
            "conc_peak": float(p.max()),
            "conc_ref": conc_ref,
            "peak_hr": peak_hr,
            "ref_hr": ref,
            "peak_err_bpm": abs(peak_hr - ref),
        })
    return pd.DataFrame(rows)


def paired_within_cell(df: pd.DataFrame, value: str):
    """Within (participant, session) x ACC-quintile: median(outage) - median(ok)."""
    diffs, recs = [], []
    for (pid, sess), g in df.groupby(["participant", "session"], sort=True):
        if g["ok"].nunique() < 2:
            continue
        try:
            g = g.assign(bin=pd.qcut(g["acc_motion"], N_ACC_BINS, labels=False,
                                     duplicates="drop"))
        except ValueError:
            continue
        for b, gb in g.groupby("bin", sort=True):
            o = gb.loc[gb["ok"], value].to_numpy(float)
            u = gb.loc[~gb["ok"], value].to_numpy(float)
            if len(o) < 5 or len(u) < 5:
                continue
            d = float(np.median(u) - np.median(o))
            diffs.append(d)
            recs.append({"participant": pid, "session": sess, "bin": int(b),
                         "n_ok": int(len(o)), "n_outage": int(len(u)),
                         "median_ok": round(float(np.median(o)), 4),
                         "median_outage": round(float(np.median(u)), 4),
                         "diff": round(d, 4),
                         "acc_motion_median": round(float(np.median(gb["acc_motion"])), 4)})
    return np.asarray(diffs, dtype=float), recs


def cluster_ci(vals: np.ndarray, groups: list[str], n_boot: int = 4000, seed: int = 20261103):
    if vals.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    gs = pd.Series(groups)
    uniq = sorted(set(groups))
    idx = {u: np.flatnonzero(gs.to_numpy() == u) for u in uniq}
    obs = float(np.mean(vals))
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx[u] for u in pick])
        boot[b] = float(np.mean(vals[take]))
    return {"n": int(vals.size), "n_participants": len(uniq),
            "mean": round(obs, 4),
            "ci95": [round(float(np.percentile(boot, 2.5)), 4),
                     round(float(np.percentile(boot, 97.5)), 4)],
            "median": round(float(np.median(vals)), 4)}


def main():
    all_rows = []
    skipped = []
    for sid in DEV:
        need = ["watch_ppg", "watch_acc", "watch_hr", "polar_hr", "events"]
        if not all((RAW / sid / FILES[f][0]).exists() for f in need):
            skipped.append(sid)
            continue
        r = per_second_rows(sid)
        print(f"  {sid}: {len(r)} seconds ({int(r['ok'].sum())} ok / {int((~r['ok']).sum())} outage)")
        all_rows.append(r)
    df = pd.concat(all_rows, ignore_index=True)
    print(f"  skipped (no Galaxy Watch files): {skipped}")

    out = {
        "state": "DEV_PPG_PULSE_DURING_HR_OUTAGE_NO_FIT",
        "window": f"causal {PPG_WIN_MS} ms PPG ending at each watch-HR row; ACC same window",
        "reference": "Polar H10 HR nearest within +/-5 s, frozen offset -32400000 ms",
        "seconds": {"total": int(len(df)), "ok": int(df["ok"].sum()),
                    "outage": int((~df["ok"]).sum()),
                    "participants": int(df["participant"].nunique())},
        "sealed_UNREAD": ["P08", "P13", "P17", "P19", "P20", "P23"],
    }

    # 1. raw (activity-confounded) contrast
    out["unmatched_within_session"] = {}
    for v in ("conc_ref", "conc_peak", "peak_err_bpm"):
        ok = df.loc[df["ok"], v].to_numpy(float)
        ug = df.loc[~df["ok"], v].to_numpy(float)
        out["unmatched_within_session"][v] = {
            "median_ok": round(float(np.median(ok)), 4),
            "median_outage": round(float(np.median(ug)), 4),
            "n_ok": int(ok.size), "n_outage": int(ug.size)}

    # 2. ACC-matched within (participant, session)
    for v in ("conc_ref", "conc_peak", "peak_err_bpm"):
        diffs, recs = paired_within_cell(df, v)
        groups = [r["participant"] for r in recs]
        out.setdefault("acc_matched_within_session", {})[v] = cluster_ci(diffs, groups)
        out["acc_matched_within_session"][v]["n_cells"] = len(recs)

    # 3. the decisive rate: could a plain FFT peak on the watch PPG have recovered HR?
    for tol in (10, 15, 20):
        r_ok = float((df.loc[df["ok"], "peak_err_bpm"] <= tol).mean())
        r_ug = float((df.loc[~df["ok"], "peak_err_bpm"] <= tol).mean())
        cells = []
        for (pid, sess), g in df.groupby(["participant", "session"], sort=True):
            if g["ok"].nunique() < 2:
                continue
            o = g.loc[g["ok"], "peak_err_bpm"].to_numpy(float)
            u = g.loc[~g["ok"], "peak_err_bpm"].to_numpy(float)
            if len(o) >= 5 and len(u) >= 5:
                cells.append((pid, float((u <= tol).mean() - (o <= tol).mean())))
        d = np.array([c[1] for c in cells], float)
        out.setdefault("recoverable_pulse_rate", {})[f"within_{tol}bpm"] = {
            "fraction_ok_seconds": round(r_ok, 4),
            "fraction_outage_seconds": round(r_ug, 4),
            "within_session_paired": cluster_ci(d, [c[0] for c in cells])}

    out["cells"] = {
        "n_participant_session_cells_used": int(
            df.groupby(["participant", "session"]).ngroups),
        "example_matched_cells": None}
    (OUT_DIR / "ppg_pulse_during_outage.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n--- unmatched (activity-confounded) ---")
    for v, d in out["unmatched_within_session"].items():
        print(f"  {v:14s} ok {d['median_ok']}  outage {d['median_outage']}")
    print("--- ACC-matched within (participant, session): median(outage) - median(ok) ---")
    for v, d in out["acc_matched_within_session"].items():
        print(f"  {v:14s} {d['mean']:+.4f} CI[{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}] "
              f"(n_cells={d['n_cells']})")
    print("--- recoverable pulse (plain FFT peak within X bpm of Polar) ---")
    for k, d in out["recoverable_pulse_rate"].items():
        print(f"  {k:16s} ok {d['fraction_ok_seconds']:.3f}  outage {d['fraction_outage_seconds']:.3f}"
              f"  paired {d['within_session_paired'].get('mean')} "
              f"CI{d['within_session_paired'].get('ci95')}")
    print(f"\nwritten: {OUT_DIR / 'ppg_pulse_during_outage.json'}")


if __name__ == "__main__":
    main()
