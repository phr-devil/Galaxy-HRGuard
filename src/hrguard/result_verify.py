"""V1.7 step 6 -- INDEPENDENT recomputation of the SEALED confirmation.

Deliberately imports NOTHING from v1_6_confirm_run / audit_raw_dev /
probe_dev_windows. Every constant, the window grid, the session parser, the
supply share, the paired log-odds deltas, the exact sign-flip enumeration and
the cluster bootstrap are re-implemented from the frozen prose definitions so
that a coding error in one implementation does not silently reproduce itself
in the other.

Compares, item by item, against evidence/v1_6/v1_6_confirmation_ledger.json.
Reports agreement AND disagreement; a mismatch is a finding, not a failure to
hide.

Run: HRGUARD_SEALED_AUTHORIZED=1 python v1_7_independent_verify.py
"""
from __future__ import annotations
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from . import paths as _paths

def _require_authorisation() -> None:
    """The held-out cohort is guard-railed even for re-computation."""
    if os.environ.get("HRGUARD_SEALED_AUTHORIZED") != "1":
        raise PermissionError(
            "independent verification of the held-out confirmation requires the "
            "same explicit authorisation (HRGUARD_SEALED_AUTHORIZED=1)")

HERE = _paths.REPO_ROOT
RAW = _paths.sealed_root()
LEDGER = _paths.frozen_dir() / "v1_6" / "v1_6_confirmation_ledger.json"
OUT = _paths.output_dir("evidence/v1_7") / "v1_7_independent_verification.json"

# ---- constants, re-typed from the frozen protocol prose (not imported) ----
WINDOW_MS = 8000
STRIDE_MS = 2000
MIN_GOOD = 4
ERR_BPM = 10.0
HR_LO, HR_HI = 40.0, 200.0
PURE_ACTIVE = {"walking", "jogging", "running"}
CLEAN_QUIET = {"baseline", "adaptation", "meditation-1", "meditation-2",
               "screen-reading", "mobile-typing"}
SEED = 20261104
N_BOOT = 20000          # larger than the runner's 8000 on purpose


def num(s):
    return pd.to_numeric(s, errors="coerce")


def read_csv(path, usecols):
    d = pd.read_csv(path, usecols=lambda c: c in usecols)
    return d


def sessions(path):
    """ENTER/EXIT pairing, first-in-first-out, only spans >= one window."""
    d = pd.read_csv(path)
    d["timestamp"] = num(d["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = []
    for _, g in d.groupby("session", sort=False):
        queue = []
        for _, r in g.iterrows():
            kind = str(r["status"]).upper()
            if kind == "ENTER":
                queue.append(int(r["timestamp"]))
            elif kind == "EXIT" and queue:
                s = queue.pop(0)
                e = int(r["timestamp"])
                if e - s >= WINDOW_MS:
                    out.append((str(r["session"]), s, e))
    return sorted(out, key=lambda x: x[1])


def build(sid, offset_ms):
    """Rebuild the window table from raw CSVs, independently."""
    ppg = read_csv(RAW / sid / "GalaxyWatch/PPG.csv", {"timestamp"})
    whr = read_csv(RAW / sid / "GalaxyWatch/HR.csv",
                   {"timestamp", "hr", "hrStatus"})
    pol = read_csv(RAW / sid / "PolarH10/HR.csv", {"phoneTimestamp", "hr"})

    ref = pd.DataFrame({
        "t": num(pol["phoneTimestamp"]).to_numpy(float) + offset_ms,
        "hr": num(pol["hr"]).to_numpy(float),
    }).dropna().sort_values("t").reset_index(drop=True)

    watch = pd.DataFrame({
        "t": num(whr["timestamp"]).to_numpy(float),
        "hr": num(whr["hr"]).to_numpy(float),
        "st": num(whr["hrStatus"]).to_numpy(float),
    }).dropna(subset=["t"]).sort_values("t").reset_index(drop=True)
    watch = watch[(watch["st"] == 1) & watch["hr"].between(HR_LO, HR_HI)]

    rt = ref["t"].to_numpy()
    wt = watch["t"].to_numpy()

    rows = []
    for sess, s0, s1 in sessions(RAW / sid / "Event.csv"):
        for t0 in range(s0, s1 - WINDOW_MS + 1, STRIDE_MS):
            t1 = t0 + WINDOW_MS
            rv = ref["hr"].to_numpy()[np.searchsorted(rt, t0, "left"):
                                      np.searchsorted(rt, t1, "left")]
            rv = rv[np.isfinite(rv) & (rv >= HR_LO) & (rv <= HR_HI)]
            hv = watch["hr"].to_numpy()[np.searchsorted(wt, t0, "left"):
                                        np.searchsorted(wt, t1, "left")]
            ref_ok = len(rv) >= MIN_GOOD
            watch_ok = len(hv) >= MIN_GOOD
            err = abs(float(np.median(hv)) - float(np.mean(rv))) \
                if (ref_ok and watch_ok) else float("nan")
            rows.append({"participant": sid, "session": sess, "start_ms": t0,
                         "ref_ok": ref_ok, "watch_ok": watch_ok,
                         "bad": bool(np.isfinite(err) and err > ERR_BPM),
                         "missing": bool(ref_ok and not watch_ok)})
    return pd.DataFrame(rows)


def supply_share(df):
    """E1 = wrong / (wrong + refused), pooled over participants.

    Refused means: a valid Polar reference exists in the window and no usable
    watch value was delivered. Windows without a valid reference enter neither
    the numerator nor the denominator.
    """
    a = df[df["watch_ok"]]
    wrong = int(a["bad"].sum())
    refused = int(df["missing"].sum())
    return wrong, refused, wrong / (wrong + refused)


def deltas(df):
    """Paired log-odds delta per participant: active minus quiet."""
    out = {}
    for p, g in df[df["watch_ok"]].groupby("participant"):
        hi = g[g["session"].isin(PURE_ACTIVE)]
        lo = g[g["session"].isin(CLEAN_QUIET)]
        if len(hi) == 0 or len(lo) == 0:
            continue
        f = lambda x: np.log((x["bad"].sum() + 0.5) / (len(x) - x["bad"].sum() + 0.5))
        out[p] = float(f(hi) - f(lo))
    return out


def signflip(d):
    """Exact one-sided sign-flip permutation p: enumerate all 2^n patterns."""
    v = np.asarray(list(d), float)
    obs = v.mean()
    hits = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(v)):
        if float((np.array(signs) * v).mean()) >= obs - 1e-12:
            hits += 1
    return hits / float(2 ** len(v))


def boot_ci(per, rng, n_boot=N_BOOT):
    """Participant-level cluster bootstrap of the pooled supply share."""
    pids = sorted(per)
    k = np.array([per[p]["wrong"] for p in pids], float)
    r = np.array([per[p]["refused"] for p in pids], float)
    idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
    ks, rs = k[idx].sum(axis=1), r[idx].sum(axis=1)
    sh = ks / (ks + rs)
    return float(np.quantile(sh, 0.025)), float(np.quantile(sh, 0.975))


def main():
    _require_authorisation()
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    pre = json.loads((_paths.frozen_dir() / "v1_6" /
                      "v1_6_sealed_preflight.json").read_text(encoding="utf-8"))
    offsets = {s: v["decision"]["selected_offset_ms"]
               for s, v in pre["per_participant"].items()
               if v["state"] == "ALIGNMENT_PASS"}

    frames = [build(s, o) for s, o in sorted(offsets.items())]
    df = pd.concat(frames, ignore_index=True)

    per = {}
    for p, g in df.groupby("participant"):
        w, r, sh = supply_share(g)
        per[p] = {"windows": int(len(g)), "correct_or_refused":
                  int(g["ref_ok"].sum()), "wrong": w, "refused": r,
                  "share": sh}

    wrong, refused, e1 = supply_share(df)
    d = deltas(df)
    rng = np.random.default_rng(SEED + 1)
    lo, hi = boot_ci(per, rng)

    got = {
        "n_evaluable": int(df["participant"].nunique()),
        "participants": sorted(per),
        "windows_total": int(len(df)),
        "n_wrong": wrong,
        "n_refused": refused,
        "E1_pooled_supply_share": e1,
        "E1_ci95": [lo, hi],
        "E2_deltas": d,
        "E2_point_estimate": float(np.mean(list(d.values()))),
        "E2_exact_signflip_p_one_sided": signflip(d.values()),
        "per_participant": per,
    }

    led_est = led["estimands"]
    led_crit = led["criteria"]
    exp = {
        "n_evaluable": led["evaluable_n"],
        "E1_pooled_supply_share": led_est["E1_pooled_supply_share"],
        "E1_ci95": led_crit["E1_pooled_supply_share"]["ci95"],
        "E2_point_estimate": led_est["E2_within_person_activity_gradient"],
        "E2_exact_signflip_p_one_sided":
            led_crit["E2_within_person_activity_gradient"]["exact_signflip_p_one_sided"],
    }

    checks = []
    def cmp(name, a, b, tol):
        ok = (abs(float(a) - float(b)) <= tol) if not isinstance(a, list) else \
             all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))
        checks.append({"quantity": name, "ledger": b, "independent": a,
                       "abs_diff": (max(abs(float(x) - float(y))
                                        for x, y in zip(a, b))
                                    if isinstance(a, list) else abs(float(a) - float(b))),
                       "tolerance": tol, "pass": bool(ok)})

    cmp("n_evaluable", got["n_evaluable"], exp["n_evaluable"], 0)
    cmp("E1_pooled_supply_share", got["E1_pooled_supply_share"],
        exp["E1_pooled_supply_share"], 1e-9)
    cmp("E2_point_estimate", got["E2_point_estimate"], exp["E2_point_estimate"], 1e-9)
    cmp("E2_exact_signflip_p_one_sided", got["E2_exact_signflip_p_one_sided"],
        exp["E2_exact_signflip_p_one_sided"], 1e-12)

    # Integers must match exactly: this is the denominator question.
    led_per = led_est["per_participant"]
    int_ok = all(got["per_participant"][p]["wrong"] == led_per[p]["n_wrong"]
                 and got["per_participant"][p]["refused"] == led_per[p]["n_refused"]
                 for p in led_per)
    checks.append({"quantity": "per_participant_counts_(wrong,refused)",
                   "ledger": {p: [led_per[p]["n_wrong"], led_per[p]["n_refused"]]
                              for p in led_per},
                   "independent": {p: [got["per_participant"][p]["wrong"],
                                       got["per_participant"][p]["refused"]]
                                   for p in per},
                   "pass": bool(int_ok)})

    # The bootstrap interval cannot be reproduced bit-for-bit with a different
    # RNG stream and n_boot; agreement is judged against Monte-Carlo error.
    lo_l, hi_l = exp["E1_ci95"]
    mc = {"ledger_ci95": [lo_l, hi_l], "independent_ci95": [lo, hi],
          "method": "independent RNG stream and n_boot=20000, so exact equality "
                    "is not expected; agreement is judged on whether the two "
                    "intervals overlap and both stay below the 0.50 benchmark",
          "overlap": bool(lo <= hi_l and lo_l <= hi),
          "both_below_0p50": bool(hi < 0.50 and hi_l < 0.50)}

    verdict = {
        "all_estimand_checks_pass": all(c["pass"] for c in checks),
        "E1_criterion_reproduced": checks[1]["pass"],
        "bootstrap_interval_consistent": mc["overlap"] and mc["both_below_0p50"],
        "ledger_status_E1": led_crit["E1_pooled_supply_share"]["status"],
        "ledger_paper_label": led["paper_label"]["label"],
    }

    doc = {"stage": "independent_verification", "implementation":
           "separate module, no import from v1_6_confirm_run / audit_raw_dev / "
           "probe_dev_windows", "checks": checks, "bootstrap": mc,
           "verdict": verdict, "independent_values": got,
           "note": "disagreements are reported, never suppressed"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=float),
                   encoding="utf-8")
    print(json.dumps({"all_estimand_checks_pass": verdict["all_estimand_checks_pass"],
                      "E1_independent": round(e1, 6),
                      "E1_ledger": round(exp["E1_pooled_supply_share"], 6),
                      "ci_independent": [round(lo, 4), round(hi, 4)],
                      "ci_ledger": [round(lo_l, 4), round(hi_l, 4)],
                      "checks": [{"q": c["quantity"], "pass": c["pass"]}
                                 for c in checks],
                      "out": str(OUT)}, indent=2, ensure_ascii=False))
    return 0 if verdict["all_estimand_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
