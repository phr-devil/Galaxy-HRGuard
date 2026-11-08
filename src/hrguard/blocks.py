"""Science-upgrade pilot, part 2b: block-level refusal accounting.

Part 2a could not form a within-block valid-vs-refused contrast for jogging
and running because those blocks are refused almost end to end. That absence
is itself the finding, so this script quantifies it directly:

  for every protocol block (participant x session instance) report
     block duration, refused seconds, refused fraction, and the reference-HR
     range over the block, separately for the refused and valid sub-periods.

No fitting, no selection. Unit of inference = participant (block-level values
are first averaged within participant, then summarised across participants).
"""
import json
import os
import numpy as np
import pandas as pd
from . import paths as _paths
_RAW = _paths.data_root()
_OUTD = _paths.output_dir("evidence/science_upgrade")

OFF_MS = -32400000
HIGH = {"jogging", "running"}
LOW = {"meditation-1", "meditation-2", "baseline", "adaptation",
       "rest-1", "rest-2", "rest-3", "rest-4", "rest-5"}


def main():
    sids = sorted(d for d in os.listdir(str(_RAW))
                  if os.path.exists(f"{_RAW}/{d}/GalaxyWatch/HR.csv"))
    blocks = []
    for sid in sids:
        w = pd.read_csv(f"{_RAW}/{sid}/GalaxyWatch/HR.csv")
        w["timestamp"] = pd.to_numeric(w["timestamp"], errors="coerce")
        w = w[w["timestamp"].notna()].copy()
        w["hr"] = pd.to_numeric(w["hr"], errors="coerce")
        w["hrStatus"] = pd.to_numeric(w["hrStatus"], errors="coerce")
        w = w.sort_values("timestamp").reset_index(drop=True)
        t = w["timestamp"].to_numpy(float)
        v = (w["hr"].to_numpy(float) > 0)
        p = pd.read_csv(f"{_RAW}/{sid}/PolarH10/HR.csv")
        pt = pd.to_numeric(p["phoneTimestamp"], errors="coerce").to_numpy(float) + OFF_MS
        ph = pd.to_numeric(p["hr"], errors="coerce").to_numpy(float)
        ok = np.isfinite(pt) & np.isfinite(ph) & (ph > 0)
        pt, ph = pt[ok], ph[ok]
        o = np.argsort(pt)
        pt, ph = pt[o], ph[o]
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
        for name, a, b in ses:
            if b - a < 5000:
                continue
            m = (t >= a) & (t <= b)
            if m.sum() < 5:
                continue
            dur = (b - a) / 1000.0
            ref = (v[m])
            refused_s = float((~ref).sum())
            # reference range over whole block and inside each state
            pm = (pt >= a) & (pt <= b)
            rows = {"subject": sid, "session": name, "block_s": dur,
                    "refused_s": refused_s, "refused_frac": refused_s / dur,
                    "n_rows": int(m.sum())}
            if pm.sum() > 5:
                rows["ref_range_block"] = float(np.ptp(ph[pm]))
            # refused sub-period reference range (contiguous, >=10 s)
            sub = t[m][~ref]
            if sub.size >= 10:
                pm2 = (pt >= sub.min()) & (pt <= sub.max())
                if pm2.sum() > 5:
                    rows["ref_range_in_refused"] = float(np.ptp(ph[pm2]))
            sub2 = t[m][ref]
            if sub2.size >= 10:
                pm3 = (pt >= sub2.min()) & (pt <= sub2.max())
                if pm3.sum() > 5:
                    rows["ref_range_in_valid"] = float(np.ptp(ph[pm3]))
            blocks.append(rows)

    b = pd.DataFrame(blocks)
    b["motion"] = np.where(b["session"].isin(HIGH), "high",
                           np.where(b["session"].isin(LOW), "low", "other"))

    out = {"status": "EXPLORATORY / NOT_INDEPENDENTLY_CONFIRMED",
           "part": "block-level refusal accounting",
           "n_blocks": int(len(b)), "n_subjects": int(b["subject"].nunique()),
           "blocks": b.to_dict("records")}

    # per-participant averages, then across participants
    pv = b.groupby(["subject", "motion"])["refused_frac"].mean().unstack()
    out["per_subject_refused_frac_by_motion"] = {
        str(i): {str(c): (None if pd.isna(v) else float(v)) for c, v in r.items()}
        for i, r in pv.iterrows()}

    def summarise(col):
        x = pv[col].dropna().to_numpy(float) if col in pv else np.array([])
        if x.size == 0:
            return {}
        return {"n_subjects": int(x.size), "mean": float(x.mean()),
                "median": float(np.median(x)), "min": float(x.min()),
                "max": float(x.max())}

    out["refused_frac_summary"] = {c: summarise(c) for c in ("high", "low", "other")
                                   if c in pv}

    hi = b[b["motion"] == "high"]
    lo = b[b["motion"] == "low"]
    out["high_motion_blocks"] = {
        "n": int(len(hi)),
        "median_block_s": float(hi["block_s"].median()) if len(hi) else None,
        "median_refused_frac": float(hi["refused_frac"].median()) if len(hi) else None,
        "frac_blocks_fully_refused": float((hi["refused_frac"] > 0.95).mean()) if len(hi) else None,
        "frac_blocks_with_any_refusal": float((hi["refused_frac"] > 0).mean()) if len(hi) else None,
        "median_ref_range": float(hi["ref_range_block"].median()) if len(hi) else None,
    }
    out["low_motion_blocks"] = {
        "n": int(len(lo)),
        "median_block_s": float(lo["block_s"].median()) if len(lo) else None,
        "median_refused_frac": float(lo["refused_frac"].median()) if len(lo) else None,
        "frac_blocks_fully_refused": float((lo["refused_frac"] > 0.95).mean()) if len(lo) else None,
        "frac_blocks_with_any_refusal": float((lo["refused_frac"] > 0).mean()) if len(lo) else None,
        "median_ref_range": float(lo["ref_range_block"].median()) if len(lo) else None,
    }
    # how much of total reference HR range is inside refused time
    out["worst_blocks"] = (hi.sort_values("refused_frac", ascending=False)
                           .head(10)[["subject", "session", "block_s", "refused_frac",
                                      "ref_range_block", "ref_range_in_refused"]]
                           .to_dict("records"))

    
    json.dump(out, open(f"{_OUTD}/pilot_blocks.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print("blocks:", len(b), "| subjects:", b["subject"].nunique())
    for k in ("high_motion_blocks", "low_motion_blocks"):
        d = out[k]
        print(f"{k}: n={d['n']} block {d['median_block_s']:.0f}s  refused frac median {d['median_refused_frac']:.2f}"
              f"  fully-refused {d['frac_blocks_fully_refused']:.2f}  any-refusal {d['frac_blocks_with_any_refusal']:.2f}"
              f"  ref range {d['median_ref_range']:.1f} bpm")
    print("per-participant refused fraction (median across participants):")
    for c, v in out["refused_frac_summary"].items():
        print(f"   {c:6} n={v['n_subjects']:2d}  median {v['median']:.3f}  range [{v['min']:.3f}, {v['max']:.3f}]")
    print("worst high-motion blocks:")
    for r in out["worst_blocks"]:
        print(f"   {r['subject']} {r['session']:10} block {r['block_s']:5.0f}s refused {100*r['refused_frac']:5.1f}% "
              f"ref range {r['ref_range_block']:.1f} bpm")


if __name__ == "__main__":
    main()
