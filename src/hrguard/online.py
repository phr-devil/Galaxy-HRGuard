"""Protocol fix: the evaluation core that every v0.7/v0.8 arm must share.

WHAT THIS FIXES
---------------
Three defects were found by the v4 audit (audit_recall_denominator.py,
audit_stratum_selection.py, audit_ppg_increment.py).  All three are evaluation-layer
defects, not modelling defects, and all three are FIXED HERE so that the CPU pilot and
every later verifier share one implementation:

  FIX-1  EVENT DENOMINATOR.
         `same_protocol_rules.py` computed event_recall as
             (deduplicated onset EPISODES detected) / (positive WINDOWS).
         The numerator and the denominator counted different things.  With 617 positive
         windows over 185 pair-space runs and 141 pilot onset episodes, this understated
         the episode recall by a factor of 4.4.  Every recall number in this project
         must now be computed with `onset_episode_events()` as the denominator and must
         state the denominator explicitly in its output.

  FIX-2  THE PRIMARY CONTRAST.
         The v0.7 rule compared M2 - B2 (full model vs ACC-only baseline).  Inside a
         stratum defined by the dispersion of the ACC score, that contrast is dominated
         by baseline weakness (corr = -0.757 between the baseline's AUROC and the
         measured "gain").  The contrast that isolates the scientific quantity of
         interest is M2 - M1, because M1 = ACC + watch-HR-state already contains ACC,
         so adding PPG is the only change.  `contrast_ppg_increment()` computes it.

  FIX-3  FOLD-INTERNAL THRESHOLD.
         The stratum threshold must never be estimated on the evaluation fold.  The
         frozen pilot scored all cells against a DEV-wide quantile, which leaks the
         evaluation fold's dispersion distribution into the split.  `fold_internal_split`
         re-derives the threshold from training cells only.

  FIX-4  CAUSAL STRATUM DEFINITION.
         Dispersion over the whole session is not available at inference time.
         `causal_dispersion()` computes it from a prefix only.

Nothing here fits anything.  The CPU pilot is the only thing that fits, and it is not
authorised yet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW_MS = 8000
LEAD_MS = 10000
STRIDE_MS = 2000
COOLDOWN_MS = 10000
HORIZON_MS = 10000

# The one pre-registered stratum threshold for the upcoming pilot.
PRE_REGISTERED_QUANTILE = 0.50

# The quantile sweep is a ROBUSTNESS family, never a search.
ROBUSTNESS_QUANTILES = (0.25, 0.40, 0.50, 0.60, 0.75)


# --------------------------------------------------------------------- episode universe
def episode_table(u: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of unreliable windows inside one (participant, session).

    `u` is the window universe and must carry the columns
    participant, session, start_ms, y_unreliable.
    """
    u = u.sort_values(["participant", "session", "start_ms"]).reset_index(drop=True)
    unrel = (u.y_unreliable > 0.5).to_numpy()
    eid = np.full(len(u), -1, dtype=int)
    cur, n = None, 0
    prev_pid, prev_sess, prev_st = None, None, None
    for i in range(len(u)):
        pid, sess, st = u.participant.iat[i], u.session.iat[i], int(u.start_ms.iat[i])
        if unrel[i]:
            if cur is None or (pid, sess) != (prev_pid, prev_sess) or (st - prev_st) > STRIDE_MS + 500:
                cur = n
                n += 1
            eid[i] = cur
        else:
            cur = None
        prev_pid, prev_sess, prev_st = pid, sess, st
    return pd.DataFrame({"participant": u.participant, "session": u.session,
                         "start_ms": u.start_ms, "episode_id": eid})


def onset_episode_events(u: pd.DataFrame, ep: pd.DataFrame | None = None) -> pd.DataFrame:
    """THE event set for every alarm metric: one row per onset episode.

    An onset episode is a run of unreliable target windows whose FIRST window has a
    history window 10 s earlier AND that history window is RELIABLE.  This is the
    frozen pilot's definition (141 episodes on DEV) and it is the only denominator
    that episode-level recall may use.

    The returned frame carries `denominator_kind = "onset_episode"`, which every
    metric below copies into its output, so no number can be reported without
    declaring what it is divided by.
    """
    if ep is None:
        ep = episode_table(u)
    key = {(r.participant, r.session, int(r.start_ms)): float(r.y_unreliable)
           for r in u.itertuples()}
    e = ep[ep.episode_id >= 0]
    first = (e.groupby(["participant", "session", "episode_id"]).start_ms.min()
             .rename("t_start").reset_index())
    nwin = (e.groupby(["participant", "session", "episode_id"]).size()
            .rename("n_windows").reset_index())
    first = first.merge(nwin, on=["participant", "session", "episode_id"])
    rows = []
    for r in first.itertuples():
        h = key.get((r.participant, r.session, int(r.t_start) - LEAD_MS))
        rows.append({**r._asdict(),
                     "has_history": h is not None,
                     "history_reliable": bool(h < 0.5) if h is not None else False})
    out = pd.DataFrame(rows)
    out = out[out.history_reliable].copy()
    return out.sort_values(["participant", "session", "t_start"]).reset_index(drop=True)


# --------------------------------------------------------------------- stratum definition
def causal_dispersion(df: pd.DataFrame, col: str = "score_B2_acc_motion",
                      prefix_frac: float = 1.0) -> dict:
    """Within-cell SD of the ACC-motion score over the cell's history windows.

    `prefix_frac = 1.0` reproduces the frozen pilot's whole-session dispersion.
    `prefix_frac < 1.0` uses only the first fraction of the session, which is what a
    deployed system could actually compute at inference time.

    Input-only: uses a feature column, never an outcome label.
    """
    out = {}
    for key, g in df.groupby(["participant", "session"]):
        g = g.sort_values("h_start")
        if prefix_frac < 1.0:
            g = g.iloc[:max(2, int(len(g) * prefix_frac))]
        out[key] = float(np.std(g[col].to_numpy(dtype=float)))
    return out


def fold_internal_split(disp: dict, train_cells, eval_cells, quantile=PRE_REGISTERED_QUANTILE):
    """FIX-3: the stratum threshold is estimated on TRAINING CELLS ONLY.

    Returns (is_low_eval_mask, threshold).  `is_low_eval_mask` is a list of booleans, one
    per evaluation cell, in the order given by `eval_cells`.
    """
    train_vals = np.array([disp[k] for k in train_cells if k in disp], dtype=float)
    if len(train_vals) == 0:
        raise ValueError("no training cells available for the stratum threshold")
    thr = float(np.quantile(train_vals, quantile))
    mask = [bool(disp.get(k, np.inf) <= thr) for k in eval_cells]
    return mask, thr


# --------------------------------------------------------------------- alarm machinery
def greedy_alarms(pid, sess, tau, score, events, hours, budget,
                  cooldown=COOLDOWN_MS, horizon=HORIZON_MS, session_strict=True):
    """Greedy top-K alarm placement with the fixed protocol.

    FIX-1 applies here: `event_recall` divides deduplicated detected EPISODES by the
    number of ONSET EPISODES in `events`, and every output field states the
    denominator kind.

    `events` must be an onset-episode frame (one row per episode).  Passing a
    positive-window frame is a protocol error and will be caught.
    """
    if len(events) and "episode_id" in events.columns:
        n_win = events.groupby(["participant", "session", "episode_id"]).size()
        if n_win.max() > 1:
            raise ValueError(
                "events frame contains repeated episode keys: this is the window-denominator "
                "bug (FIX-1).  Pass onset_episode_events(), not the positive windows.")
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    k_budget = {p: int(round(budget * h)) for p, h in hours.items()}
    ev_pid = events.participant.to_numpy()
    ev_sess = events.session.to_numpy()
    ev_t = events.t_start.to_numpy(dtype=float)
    ev_key = list(zip(events.participant, events.session, events.episode_id))
    used, placed, det, leads = {}, {}, set(), []
    n_fa, redundant, multi = 0, 0, 0
    for i in order:
        p, s, t = pid[i], sess[i], float(tau[i])
        if used.get(p, 0) >= k_budget.get(p, 0):
            continue
        key = (p, s) if session_strict else p
        near = placed.get(key)
        if near and min(abs(t - x) for x in near) < cooldown:
            continue
        used[p] = used.get(p, 0) + 1
        placed.setdefault(key, []).append(t)
        m = (ev_pid == p) & (ev_t >= t) & (ev_t <= t + horizon)
        if session_strict:
            m &= (ev_sess == s)
        if m.any():
            if m.sum() > 1:
                multi += 1
            j = int(np.flatnonzero(m)[np.argmin(ev_t[m])])
            leads.append(float(ev_t[j] - t))
            if ev_key[j] in det:
                redundant += 1
            det.add(ev_key[j])
        else:
            n_fa += 1
    per_rec = {}
    for p, g in events.groupby("participant"):
        per_rec[p] = float(sum(1 for r in g.itertuples()
                               if (r.participant, r.session, int(r.episode_id)) in det) / len(g))
    per_fa = {}
    for p, h in hours.items():
        tp = sum(1 for k in det if k[0] == p)
        per_fa[p] = float((used.get(p, 0) - tp) / max(h, 1e-9))
    n_tp = len(det)
    return {
        "denominator_kind": "onset_episode",
        "n_onset_episodes": int(len(events)),
        "alarms": int(sum(used.values())),
        "events_detected": int(n_tp),
        "episode_recall": float(n_tp / max(len(events), 1)),
        "macro_episode_recall": float(np.mean(list(per_rec.values()))) if per_rec else None,
        "alarm_precision": float(n_tp / max(sum(used.values()), 1)),
        "false_alarms": int(n_fa),
        "macro_false_alarms_per_hour": float(np.mean(list(per_fa.values()))),
        "median_lead_ms": float(np.median(leads)) if leads else None,
        "n_lead_ge_8s": int(sum(1 for x in leads if x >= 8000)),
        "redundant_alarms_on_detected_episode": int(redundant),
        "alarms_matching_multiple_episodes": int(multi),
        "per_participant_recall": {k: v for k, v in sorted(per_rec.items())},
    }


# --------------------------------------------------------------------- AUROC contrasts
def auroc(y, s):
    """Tie-corrected AUROC (Mann-Whitney U with mid-ranks)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    sm = np.zeros(len(u))
    np.add.at(sm, inv, ranks)
    ranks = (sm / c)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cell_aurocs(df, col, min_pos=3, min_neg=3, min_rows=12):
    """Per-(participant, session) AUROC for one score column."""
    out = {}
    for key, g in df.groupby(["participant", "session"]):
        y = g.y.to_numpy()
        if min_pos <= y.sum() <= len(y) - min_neg and len(g) >= min_rows:
            a = auroc(y, g[col].to_numpy())
            if np.isfinite(a):
                out[key] = a
    return out


def contrast_ppg_increment(cells_m1, cells_m2, keys):
    """FIX-2: the PPG-only increment, M2 - M1, over a given set of cells.

    M1 already contains the ACC and watch-HR-state blocks, so M2 - M1 isolates the PPG
    block and is not confounded by baseline weakness (the v0.7 M2 - B2 contrast is).
    """
    common = [k for k in keys if k in cells_m1 and k in cells_m2]
    if not common:
        return None
    d = {k: cells_m2[k] - cells_m1[k] for k in common}
    return {"n_cells": len(common), "n_subjects": len({k[0] for k in common}),
            "mean_M1": float(np.mean([cells_m1[k] for k in common])),
            "mean_M2": float(np.mean([cells_m2[k] for k in common])),
            "increment": float(np.mean(list(d.values()))),
            "dmap": d}


def participant_cluster_bootstrap(dmap, n_boot=8000, seed=20261103):
    """Participant-clustered bootstrap of the mean cell difference."""
    if not dmap:
        return float("nan"), np.array([])
    by = {}
    for k, v in dmap.items():
        by.setdefault(k[0], []).append(v)
    pids = sorted(by)
    obs = float(np.mean([v for p in pids for v in by[p]]))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        pick = rng.choice(pids, size=len(pids), replace=True)
        s = []
        for p in pick:
            s.extend(by[p])
        if s:
            boot.append(np.mean(s))
    return obs, np.array(boot)


def participant_consistency(dmap, cells_a, cells_b):
    """How many participants improve, computed on the PARTICIPANT-MEAN level."""
    by = {}
    for k, v in dmap.items():
        by.setdefault(k[0], []).append(v)
    n_better = n_worse = 0
    for p in sorted(by):
        pa = float(np.mean([cells_a[k] for k in dmap if k[0] == p]))
        pb = float(np.mean([cells_b[k] for k in dmap if k[0] == p]))
        if pb > pa:
            n_better += 1
        elif pb < pa:
            n_worse += 1
    return n_better, n_worse, len(by)
