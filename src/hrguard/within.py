"""Science-upgrade pilot, part 2: remove the activity confound.

Part 1 showed that refusal gaps co-occur with jogging/running, so a
duration-matched control drawn from valid periods is confounded with
activity. This script does the only comparison that removes that confound:

  within the SAME session instance (same participant, same protocol block),
  compare reference-HR dynamics in refused sub-periods against the valid
  sub-periods of that same block.

Also reports the coverage-error trade-off as a function of gap duration, and
counts counterexamples (long gaps in low-motion sessions).

Everything is descriptive; inference unit = participant; no fitting.
"""
import json
import os
import numpy as np
import pandas as pd
from . import paths as _paths
_RAW = _paths.data_root()
_OUTD = _paths.output_dir("evidence/science_upgrade")

OFF_MS = -32400000
WIN_S = 60.0
MIN_SEG_S = 30.0


def load(sid):
    w = pd.read_csv(f"{_RAW}/{sid}/GalaxyWatch/HR.csv")
    w["timestamp"] = pd.to_numeric(w["timestamp"], errors="coerce")
    w = w[w["timestamp"].notna()].copy()
    for c in ("hr", "hrStatus"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.sort_values("timestamp").reset_index(drop=True)
    p = pd.read_csv(f"{_RAW}/{sid}/PolarH10/HR.csv")
    pt = pd.to_numeric(p["phoneTimestamp"], errors="coerce").to_numpy(float) + OFF_MS
    ph = pd.to_numeric(p["hr"], errors="coerce").to_numpy(float)
    ok = np.isfinite(pt) & np.isfinite(ph) & (ph > 0)
    pt, ph = pt[ok], ph[ok]
    o = np.argsort(pt)
    e = pd.read_csv(f"{_RAW}/{sid}/Event.csv")
    e["timestamp"] = pd.to_numeric(e["timestamp"], errors="coerce")
    e = e[e["timestamp"].notna()].sort_values("timestamp")
    ses, open_ = [], {}
    for _, r in e.iterrows():
        s, st = str(r["session"]), str(r["status"]).upper()
        if st == "ENTER":
            open_[s] = float(r["timestamp"])
        elif st == "EXIT" and s in open_:
            ses.append((s, open_[s], float(r["timestamp"])))
    return w, pt[o], ph[o], ses


def dyn(pt, ph, t0, t1):
    """Reference-HR dynamics over [t0,t1] ms: absolute range and max slope."""
    m = (pt >= t0) & (pt <= t1)
    if m.sum() < 5:
        return None
    v = ph[m]
    tt = pt[m] / 1000.0
    rng = float(np.ptp(v))
    if v.size > 2:
        sl = np.abs(np.diff(v) / np.maximum(np.diff(tt), 1e-6))
        slope = float(np.percentile(sl, 95))
    else:
        slope = 0.0
    return {"n": int(v.size), "range": rng, "slope_p95": slope,
            "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0}


def main():
    sids = sorted(d for d in os.listdir(str(_RAW))
                  if os.path.exists(f"{_RAW}/{d}/GalaxyWatch/HR.csv"))
    within, counter, buckets = [], [], {}
    per_subj = []

    for sid in sids:
        w, pt, ph, ses = load(sid)
        t = w["timestamp"].to_numpy(float)
        hr = w["hr"].to_numpy(float)
        valid = hr > 0
        rows = []
        for name, a, b in ses:
            if b - a < MIN_SEG_S * 1000:
                continue
            # sub-sample the block on a fixed 1 s grid
            grid = np.arange(a, b + 1, 1000.0)
            idx = np.searchsorted(t, grid).clip(0, len(t) - 1)
            ok = np.abs(t[idx] - grid) <= 1500
            grid, idx = grid[ok], idx[ok]
            if grid.size < MIN_SEG_S:
                continue
            v = valid[idx]
            for want, label in ((True, "valid"), (False, "refused")):
                sel = grid[v == want]
                if sel.size < MIN_SEG_S:
                    continue
                # contiguous run of at least WIN_S inside this state
                brk = np.where(np.diff(sel) > 2000)[0]
                segs = np.split(sel, brk + 1)
                for s in segs:
                    if s.size >= WIN_S:
                        d = dyn(pt, ph, s[0], s[0] + WIN_S * 1000)
                        if d:
                            d.update({"sid": sid, "session": name, "state": label,
                                      "window_s": WIN_S})
                            rows.append(d)
        if not rows:
            continue
        wd = pd.DataFrame(rows)
        for name, g in wd.groupby("session"):
            gv = g[g["state"] == "valid"]
            gr = g[g["state"] == "refused"]
            if len(gv) and len(gr):
                within.append({
                    "sid": sid, "session": name,
                    "n_valid_w": int(len(gv)), "n_refused_w": int(len(gr)),
                    "range_valid": float(gv["range"].median()),
                    "range_refused": float(gr["range"].median()),
                    "ratio": float(gr["range"].median() / max(gv["range"].median(), 1e-9)),
                })
        # counterexamples: long refused windows in low-motion sessions
        low = {"meditation-1", "meditation-2", "baseline", "adaptation",
               "rest-1", "rest-2", "rest-3", "rest-4", "rest-5"}
        for _, r in wd[(wd["state"] == "refused")].iterrows():
            if r["session"] in low:
                counter.append({"sid": sid, "session": r["session"],
                                "ref_range": r["range"]})
        # per-subject aggregate of within-session ratio
        rs = [x["ratio"] for x in within if x["sid"] == sid]
        per_subj.append({"subject": sid, "n_sessions_compared": len(rs),
                         "median_within_ratio": float(np.median(rs)) if rs else None})

    # coverage-error trade-off by gap duration (from part 1 evidence)
    p1 = json.load(open(f"{_OUTD}/pilot_dev.json", encoding="utf-8"))
    g = pd.DataFrame(p1["gaps"])
    g["bucket"] = pd.cut(g["dur"], [10, 30, 60, 120, 300],
                         labels=["10-30s", "30-60s", "60-120s", "120s+"])
    for lab, gg in g.groupby("bucket", observed=True):
        buckets[str(lab)] = {
            "n": int(len(gg)),
            "median_dur": float(gg["dur"].median()),
            "median_abs_delta": float(gg["delta"].abs().median()),
            "median_mae_locf": float(gg["mae_locf"].median()),
            "p90_mae_locf": float(np.percentile(gg["mae_locf"], 90)),
            "median_mae_linear": float(gg["mae_linear"].median()),
            "max_abs_delta": float(gg["delta"].abs().max()),
        }

    wdf = pd.DataFrame(within)
    out = {
        "status": "EXPLORATORY / NOT_INDEPENDENTLY_CONFIRMED",
        "part": "activity-confound-removed within-session contrast",
        "window_s": WIN_S, "min_segment_s": MIN_SEG_S,
        "within_session": wdf.to_dict("records") if len(wdf) else [],
        "within_summary": ({
            "n_comparisons": int(len(wdf)),
            "n_subjects": int(wdf["sid"].nunique()),
            "median_ratio": float(wdf["ratio"].median()),
            "p25": float(wdf["ratio"].quantile(0.25)),
            "p75": float(wdf["ratio"].quantile(0.75)),
            "frac_ratio_gt1": float((wdf["ratio"] > 1).mean()),
        } if len(wdf) else {}),
        "per_subject_within_ratio": per_subj,
        "counterexamples_low_motion_gaps": counter,
        "n_counterexamples": len(counter),
        "duration_buckets": buckets,
        "total_gap_seconds": float(g["dur"].sum()),
        "n_gaps": int(len(g)),
    }
    
    with open(f"{_OUTD}/pilot_within_session.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("within-session comparisons:", len(wdf), "| subjects:", wdf["sid"].nunique() if len(wdf) else 0)
    if len(wdf):
        print("  median ratio refused/valid range: %.2f  [p25 %.2f, p75 %.2f]  frac>1 %.2f"
              % (out["within_summary"]["median_ratio"], out["within_summary"]["p25"],
                 out["within_summary"]["p75"], out["within_summary"]["frac_ratio_gt1"]))
        print("  by session:")
        for s, gg in wdf.groupby("session"):
            print(f"    {s:22} n={len(gg):3d}  median ratio {gg['ratio'].median():.2f}")
    print("counterexamples (long refusal in low-motion sessions):", len(counter))
    for c in counter[:8]:
        print("   ", c["sid"], c["session"], "ref range %.1f bpm" % c["ref_range"])
    print("duration buckets:")
    for k, v in buckets.items():
        print(f"   {k:8} n={v['n']:3d}  med dur {v['median_dur']:6.0f}s  |delta| {v['median_abs_delta']:5.1f}  "
              f"LOCF MAE {v['median_mae_locf']:5.1f} (p90 {v['p90_mae_locf']:5.1f})  linear {v['median_mae_linear']:5.1f}")


if __name__ == "__main__":
    main()
