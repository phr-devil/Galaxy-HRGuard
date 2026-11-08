"""Galaxy-HRGuard science-upgrade pilot (DEV-only, zero-fit, exploratory).

Pre-specified analysis plan (written before computing any gap statistic):
  * Unit of inference = participant.
  * Watch timeline = 1 Hz rows in GalaxyWatch/HR.csv, ordered by `timestamp`.
  * Refusal event = maximal run of consecutive rows with hr == 0.
  * Reference = PolarH10 HR at the frozen clock offset (-32400 s), linear
    interpolation onto the watch timeline. The offset is NOT selected by
    comparing HR values.
  * A gap is "evaluable" if duration >= 10 s and Polar covers >= 90% of it.
  * Recovery schemes, all pre-declared:
      LOCF   causal, uses only the last valid value before the gap
      MEAN   offline, mean of boundary values
      LINEAR offline, linear ramp between boundary values
  * Control windows: same duration, drawn from valid-HR periods, same count
    and matched on the participant (not on HR level).

No model fitting, no parameter search, no stratified re-selection.
"""
import json
import os
import numpy as np
import pandas as pd
from . import paths as _paths
_RAW = _paths.data_root()
_OUTD = _paths.output_dir("evidence/science_upgrade")

OFF_MS = -32400000
MIN_GAP_S = 10.0
MIN_POLAR_COVER = 0.90
N_CONTROL_PER_GAP = 3
SEED = 20261104


def load_watch(sid):
    w = pd.read_csv(f"{_RAW}/{sid}/GalaxyWatch/HR.csv")
    w["timestamp"] = pd.to_numeric(w["timestamp"], errors="coerce")
    w = w[w["timestamp"].notna()].copy()
    for c in ("hr", "hrStatus", "dataReceived"):
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w.sort_values("timestamp").reset_index(drop=True)
    return w


def load_polar(sid):
    p = pd.read_csv(f"{_RAW}/{sid}/PolarH10/HR.csv")
    t = pd.to_numeric(p["phoneTimestamp"], errors="coerce").to_numpy(float) + OFF_MS
    h = pd.to_numeric(p["hr"], errors="coerce").to_numpy(float)
    ok = np.isfinite(t) & np.isfinite(h) & (h > 0)
    t, h = t[ok], h[ok]
    o = np.argsort(t)
    return t[o], h[o]


def load_sessions(sid):
    f = f"{_RAW}/{sid}/Event.csv"
    if not os.path.exists(f):
        return []
    e = pd.read_csv(f)
    e["timestamp"] = pd.to_numeric(e["timestamp"], errors="coerce")
    e = e[e["timestamp"].notna()].sort_values("timestamp")
    out, open_ = [], {}
    for _, r in e.iterrows():
        s, st = str(r["session"]), str(r["status"]).upper()
        if st == "ENTER":
            open_[s] = float(r["timestamp"])
        elif st == "EXIT" and s in open_:
            out.append((s, open_[s], float(r["timestamp"])))
    return out


def session_at(sessions, t0, t1):
    """Dominant session by overlap with [t0,t1] (ms)."""
    best, bestov = None, 0.0
    for s, a, b in sessions:
        ov = max(0.0, min(b, t1) - max(a, t0))
        if ov > bestov:
            best, bestov = s, ov
    return best


def runs_of(mask):
    chg = np.diff(np.r_[0, mask.astype(int), 0])
    return np.where(chg == 1)[0], np.where(chg == -1)[0]


def main():
    rng = np.random.default_rng(SEED)
    sids = sorted(d for d in os.listdir(str(_RAW))
                  if os.path.exists(f"{_RAW}/{d}/GalaxyWatch/HR.csv"))
    per, agree, all_gaps = [], [], []

    for sid in sids:
        w = load_watch(sid)
        t = w["timestamp"].to_numpy(float)
        hr = w["hr"].to_numpy(float)
        valid = hr > 0
        pol_t, pol_h = load_polar(sid)
        ses = load_sessions(sid)

        # --- reference validity: agreement where both devices report ---
        if len(pol_t) > 100:
            ref = np.interp(t, pol_t, pol_h, left=np.nan, right=np.nan)
            m = valid & np.isfinite(ref)
            if m.sum() > 100:
                agree.append({
                    "subject": sid, "n": int(m.sum()),
                    "mae": float(np.mean(np.abs(hr[m] - ref[m]))),
                    "bias": float(np.mean(hr[m] - ref[m])),
                    "corr": float(np.corrcoef(hr[m], ref[m])[0, 1]),
                })

        # --- timeline accounting (seconds, real time) ---
        span_s = (t[-1] - t[0]) / 1000.0
        dt = np.diff(t) / 1000.0
        lost_rows_s = float(dt[dt > 2.0].sum() - (dt > 2.0).sum())  # excess beyond 1 Hz
        blind_s = float((~valid).sum())
        e_s = 100.0 * blind_s / max(span_s, 1e-9)

        # --- refusal events ---
        s, e = runs_of(~valid)
        gaps = []
        for a, b in zip(s, e):
            d = (t[b - 1] - t[a]) / 1000.0
            codes = sorted(set(w["hrStatus"].iloc[a:b].tolist()))
            # reference inside
            mask = (pol_t >= t[a]) & (pol_t <= t[b - 1])
            cover = (t[b - 1] - t[a]) > 0 and float(mask.sum() * 1000.0 / (t[b - 1] - t[a] + 1000.0))
            gaps.append({"sid": sid, "a": int(a), "b": int(b), "t0": float(t[a]),
                         "t1": float(t[b - 1]), "dur": d, "codes": codes,
                         "polar_cover": cover,
                         "session": session_at(ses, t[a], t[b - 1])})

        ev = [g for g in gaps if g["dur"] >= MIN_GAP_S and g["polar_cover"] >= MIN_POLAR_COVER]
        dur = np.array([g["dur"] for g in gaps]) if gaps else np.array([0.0])

        # --- reference dynamics inside evaluable gaps vs matched controls ---
        rec_rows = []
        for g in ev:
            i0, i1 = g["a"], g["b"]
            pre_t = t[i0] - 5000
            post_t = t[i1] + 5000
            pre = pol_h[(pol_t >= pre_t) & (pol_t < t[i0])]
            post = pol_h[(pol_t > t[i1]) & (pol_t <= post_t)]
            inside = np.interp(np.arange(t[i0], t[i1] + 1, 1000.0), pol_t, pol_h,
                               left=np.nan, right=np.nan)
            if len(pre) == 0 or len(post) == 0 or not np.isfinite(inside).any():
                continue
            pre_m, post_m = float(np.mean(pre)), float(np.mean(post))
            inside = inside[np.isfinite(inside)]
            delta = post_m - pre_m
            exc = float(np.max(np.abs(inside - pre_m)))
            # recovery schemes, error against the true reference inside
            locf = np.full_like(inside, pre_m)
            lin = np.linspace(pre_m, post_m, len(inside))
            mean_ = np.full_like(inside, (pre_m + post_m) / 2.0)
            rec_rows.append({
                "sid": sid, "t0": g["t0"], "dur": g["dur"], "session": g["session"],
                "pre": pre_m, "post": post_m, "delta": delta, "excursion": exc,
                "ref_range": float(np.ptp(inside)),
                "mae_locf": float(np.mean(np.abs(locf - inside))),
                "mae_linear": float(np.mean(np.abs(lin - inside))),
                "mae_mean": float(np.mean(np.abs(mean_ - inside))),
            })
        all_gaps.extend(rec_rows)

        # --- matched control windows from valid-HR periods ---
        ctrl = []
        if rec_rows:
            vt = t[valid]
            for g in ev:
                d = g["dur"]
                n = int(d) + 1
                for _ in range(N_CONTROL_PER_GAP):
                    if len(vt) < n + 2:
                        break
                    j = int(rng.integers(1, len(vt) - n - 1))
                    a_t, b_t = vt[j], vt[j + n]
                    if b_t - a_t > 3 * d * 1000:
                        continue
                    pre = pol_h[(pol_t >= a_t - 5000) & (pol_t < a_t)]
                    post = pol_h[(pol_t > b_t) & (pol_t <= b_t + 5000)]
                    ins = np.interp(np.arange(a_t, b_t + 1, 1000.0), pol_t, pol_h,
                                    left=np.nan, right=np.nan)
                    ins = ins[np.isfinite(ins)]
                    if len(pre) == 0 or len(post) == 0 or ins.size == 0:
                        continue
                    pm, qm = float(np.mean(pre)), float(np.mean(post))
                    ctrl.append({"sid": sid, "dur": float((b_t - a_t) / 1000.0),
                                 "delta": qm - pm,
                                 "excursion": float(np.max(np.abs(ins - pm))),
                                 "ref_range": float(np.ptp(ins))})
        per.append({
            "subject": sid, "span_s": span_s, "rows": int(len(t)),
            "n_refusal_events": int(len(gaps)),
            "n_evaluable_gaps": int(len(rec_rows)),
            "blind_s": blind_s, "blind_pct": e_s,
            "record_gap_rows": int((dt > 2).sum()), "record_gap_s": lost_rows_s,
            "gap_med_s": float(np.median(dur)), "gap_p90_s": float(np.percentile(dur, 90)),
            "gap_max_s": float(dur.max()),
            "gap_total_s": float(dur.sum()),
            "n_gap_ge10": int((dur >= 10).sum()), "n_gap_ge60": int((dur >= 60).sum()),
            "ctrl": ctrl,
        })

    # ---------------- aggregate (participant level) ----------------
    ev = pd.DataFrame(all_gaps)
    ag = pd.DataFrame(agree)
    pp = pd.DataFrame([{k: v for k, v in r.items() if k != "ctrl"} for r in per])
    ctrl = pd.DataFrame([c for r in per for c in r["ctrl"]])

    def pm(x):
        x = np.asarray([v for v in x if np.isfinite(v)], float)
        if x.size == 0:
            return dict(n=0)
        return dict(n=int(x.size), mean=float(x.mean()),
                    sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                    median=float(np.median(x)),
                    lo=float(np.percentile(x, 2.5)), hi=float(np.percentile(x, 97.5)))

    out = {
        "status": "EXPLORATORY / NOT_INDEPENDENTLY_CONFIRMED",
        "design": {"offset_ms": OFF_MS, "min_gap_s": MIN_GAP_S,
                   "min_polar_cover": MIN_POLAR_COVER,
                   "n_control_per_gap": N_CONTROL_PER_GAP, "seed": SEED,
                   "unit_of_inference": "participant",
                   "offset_selected_by_hr": False, "models_fitted": 0},
        "cohort": {"n": int(len(pp)), "subjects": pp["subject"].tolist()},
        "reference_agreement": {
            "per_subject": ag.to_dict("records"),
            "median_mae": float(ag["mae"].median()) if len(ag) else None,
            "median_corr": float(ag["corr"].median()) if len(ag) else None,
            "median_bias": float(ag["bias"].median()) if len(ag) else None,
        },
        "per_subject": pp.to_dict("records"),
        "blind_pct": pm(pp["blind_pct"]),
        "gap_max_s": pm(pp["gap_max_s"]),
        "gap_total_s": pm(pp["gap_total_s"]),
        "events_ge10_per_subject": pm(pp["n_gap_ge10"]),
        "events_ge60_per_subject": pm(pp["n_gap_ge60"]),
        "gaps": ev.to_dict("records"),
        "controls": ctrl.to_dict("records"),
    }

    if len(ev) and len(ctrl):
        out["contrast"] = {}
        for k in ("delta", "excursion", "ref_range"):
            d1 = ev[k].to_numpy(float)
            d0 = ctrl[k].to_numpy(float)
            out["contrast"][k] = {
                "gap": {"n": int(d1.size), "mean": float(d1.mean()),
                        "median": float(np.median(d1)),
                        "mean_abs": float(np.mean(np.abs(d1)))},
                "control": {"n": int(d0.size), "mean": float(d0.mean()),
                            "median": float(np.median(d0)),
                            "mean_abs": float(np.mean(np.abs(d0)))},
                "ratio_mean_abs": float(np.mean(np.abs(d1)) / max(np.mean(np.abs(d0)), 1e-9)),
            }
        out["recovery"] = {}
        for k, lab in (("mae_locf", "LOCF_causal"), ("mae_mean", "boundary_mean_offline"),
                       ("mae_linear", "linear_offline")):
            v = ev[k].to_numpy(float)
            out["recovery"][lab] = {"n": int(v.size), "median_mae": float(np.median(v)),
                                    "mean_mae": float(v.mean()),
                                    "p90_mae": float(np.percentile(v, 90)),
                                    "max_mae": float(v.max())}
        # per-subject recovery + excursion (participant level)
        pv = []
        for sid, g in ev.groupby("sid"):
            pv.append({"subject": sid, "n_gaps": int(len(g)),
                       "median_dur_s": float(g["dur"].median()),
                       "max_dur_s": float(g["dur"].max()),
                       "median_abs_delta": float(g["delta"].abs().median()),
                       "max_excursion": float(g["excursion"].max()),
                       "median_mae_locf": float(g["mae_locf"].median()),
                       "median_mae_linear": float(g["mae_linear"].median())})
        out["per_subject_gap_stats"] = pv

    # session attribution of total blind seconds
    if len(ev):
        by = ev.groupby("session")["dur"].agg(["count", "sum"]).sort_values("sum", ascending=False)
        tot = float(ev["dur"].sum())
        out["blind_seconds_by_session"] = [
            {"session": str(i), "n_gaps": int(r["count"]), "seconds": float(r["sum"]),
             "share": float(r["sum"] / tot)} for i, r in by.iterrows()]

    
    with open(f"{_OUTD}/pilot_dev.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    o = main()
    print("cohort n =", o["cohort"]["n"])
    print("reference agreement: median MAE %.2f bpm, median r %.3f, bias %.2f"
          % (o["reference_agreement"]["median_mae"], o["reference_agreement"]["median_corr"],
             o["reference_agreement"]["median_bias"]))
    for k in ("blind_pct", "gap_max_s", "gap_total_s", "events_ge10_per_subject",
              "events_ge60_per_subject"):
        d = o[k]
        print(f"{k:26} mean {d.get('mean', float('nan')):8.2f}  median {d.get('median', float('nan')):8.2f}  range [{d.get('lo', 0):.2f}, {d.get('hi', 0):.2f}]")
    if "contrast" in o:
        for k, v in o["contrast"].items():
            print(f"{k:12} gap mean|.| {v['gap']['mean_abs']:7.3f}  control mean|.| {v['control']['mean_abs']:7.3f}  ratio {v['ratio_mean_abs']:5.2f}")
    if "recovery" in o:
        for k, v in o["recovery"].items():
            print(f"{k:24} median MAE {v['median_mae']:6.2f}  p90 {v['p90_mae']:6.2f}  max {v['max_mae']:7.2f}  (n={v['n']})")
    if "blind_seconds_by_session" in o:
        print("blind seconds by session:")
        for r in o["blind_seconds_by_session"][:8]:
            print(f"   {r['session']:16} {r['seconds']:7.0f}s  {100*r['share']:5.1f}%  ({r['n_gaps']} gaps)")
