"""Galaxy-HRGuard v0.5: DEV feature engineering and provenance-gated opt-in pilot.

No download, fit, or sealed-set access occurs on import. Use audit_raw_dev,
probe_dev_windows and extract_checked_dev before requesting authorized fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend, welch
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SUBJECTS = [f"P{i:02d}" for i in range(1, 25)]
SEAL_N = 6
SEED = "GalaxyHRGuard-v0.1-20261103"
# A public, fixed ID-only assignment; no label access during allocation.
SEAL = frozenset(sorted(SUBJECTS, key=lambda s: hashlib.sha256((SEED + s).encode()).hexdigest())[:SEAL_N])
DEV = frozenset(SUBJECTS) - SEAL
FEATS = ("ppg_concentration", "ppg_entropy", "ppg_hr_disagree", "acc_motion",
         "acc_ppg_overlap", "vendor_hr_std", "vendor_hr_delta", "valid_hr_fraction", "ppg_peak_exists")


def norm_spectrum(x, fs, nfft=512):
    x = np.asarray(x, dtype=float)
    if x.size < fs * 4 or not np.isfinite(x).all() or np.std(x) < 1e-9:
        return None
    f, p = welch(detrend(x), fs=fs, nperseg=min(len(x), 200), nfft=nfft)
    ok = (f >= 0.67) & (f <= 3.33)  # 40-200 bpm
    if not ok.any() or p[ok].sum() <= 1e-15:
        return None
    p = p[ok]
    return f[ok], p / p.sum()


def signal_features(ppg, acc, watch_hr):
    """Legal features: PPG, wrist ACC, watch HR available by window end."""
    if len(ppg) != 200 or len(acc) != 200 or not np.isfinite(watch_hr):
        return None
    ps = norm_spectrum(ppg.flatten(), 25)
    # Flat/no-periodicity PPG must be *retained* as a potential failure,
    # not silently deleted from the target population.
    if ps is None:
        ppg_peak_hr, concentrated, ent, ppg_peak_exists = np.nan, 0.0, 1.0, 0.0
    else:
        f, p = ps
        peak_idx = np.argmax(p)
        ppg_peak_hr = 60.0 * f[peak_idx]
        concentrated = float(np.max(p))
        ent = float(-np.sum(p * np.log(np.maximum(p, 1e-12))) / np.log(len(p)))
        ppg_peak_exists = 1.0
    # Remove mean gravity vector before motion energy; units are m/s^2.
    dyn = acc - acc.mean(axis=0, keepdims=True)
    motion = float(np.sqrt(np.mean(np.sum(dyn ** 2, axis=1))))
    motion_spectra = []
    for j in range(3):
        q = norm_spectrum(dyn[:, j], 25)
        if q is not None and ps is not None:
            motion_spectra.append(q[1])
    overlap = float(np.max(np.minimum(p, np.mean(motion_spectra, axis=0)))) if motion_spectra else 0.0
    return {"ppg_concentration": concentrated,
            "ppg_entropy": ent,
            "ppg_peak_exists": ppg_peak_exists,
            "ppg_hr_disagree": float(abs(ppg_peak_hr - watch_hr)) if ppg_peak_exists else 200.0,
            "acc_motion": motion, "acc_ppg_overlap": overlap}


def sha256_file(path: Path):
    dig = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def risk_coverage(y, err, score, fractions=(1.0, 0.8, 0.6)):
    """Tie-aware expected risk-coverage at exact retention count.

    An uninformative constant score must NOT receive favorable 80% MAE
    by retaining whichever subject appears first in the input CSV.
    """
    y, err, score = map(lambda z: np.asarray(z, dtype=float), (y, err, score))
    if not len(score) or not np.isfinite(score).all():
        raise ValueError("Empty/nonfinite risk scores")
    out = {}
    for fraction in fractions:
        k = max(1, int(np.floor(len(score)*fraction)))
        cutoff = np.partition(score, k-1)[k-1]
        strict = score < cutoff
        equal = score == cutoff
        n_strict = int(np.sum(strict));n_equal = int(np.sum(equal))
        weights = strict.astype(float) + equal.astype(float) * ((k-n_strict) / n_equal)
        out[f"keep_{fraction:g}"] = {"N": k,
            "bad10_rate": float(np.dot(weights,y)/k),
            "MAE_bpm": float(np.dot(weights,err)/k)}
    return out



def group_macro_risk_coverage(y, err, score, groups, fractions=(1.0, 0.8, 0.6)):
    """Equal-subject selective-reporting evaluation, with tie-aware within-subject retention.

    Unlike pooled coverage, this holds retention at the same target fraction *within*
    each participant, so one large/easy participant cannot hide poor performance
    on smaller participants. This is descriptive DEV evidence, not a significance test.
    """
    y, err, score, groups = (np.asarray(z) for z in (y, err, score, groups))
    if not len(y) or not (len(y)==len(err)==len(score)==len(groups)):
        raise ValueError('Invalid group-macro input lengths')
    if pd.isna(groups).any():
        raise ValueError('Unknown participant in group-macro evaluation')
    per_subject=[]
    for gid in np.unique(groups):
        idx=groups==gid
        per_subject.append(risk_coverage(y[idx],err[idx],score[idx],fractions=fractions))
    out={}
    for frac in fractions:
        key=f'keep_{frac:g}'
        out[key]={'subjects':len(per_subject),
                  'macro_bad10_rate':float(np.mean([m[key]['bad10_rate'] for m in per_subject])),
                  'macro_MAE_bpm':float(np.mean([m[key]['MAE_bpm'] for m in per_subject]))}
    return out

def pilot(csv: Path, output: Path, *, receipt: Path | None = None,
          audit: Path | None = None, probe: Path | None = None,
          preflight: Path | None = None, alignment: Path | None = None):
    """Opt-in CPU fitting on DEVELOPMENT ONLY, with true held-out subject CV."""
    df = pd.read_csv(csv)
    req = {"participant", "bad10", "error_bpm", *FEATS}
    if not req.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(req-set(df.columns))}")
    if not set(df.participant.unique()).issubset(DEV):
        raise PermissionError("Input references sealed or unrecognized subject; refusing to fit")
    if not all(v is not None for v in (receipt, audit, probe, preflight, alignment)):
        raise PermissionError("Refusing arbitrary CSV fitting: require audited receipt, preflight and lag sensitivity reports")
    from .preflight import load_verified_evidence
    verified,_ = load_verified_evidence(csv, receipt, audit, probe)
    if not verified.equals(df):
        raise ValueError('Feature CSV no longer equals verified evidence')
    pre=json.loads(preflight.read_text(encoding='utf-8'))
    lag=json.loads(alignment.read_text(encoding='utf-8'))
    source_hashes={name:sha256_file(path) for name,path in {
        'features':csv, 'receipt':receipt, 'audit':audit, 'probe':probe}.items()}
    if pre.get('state')!='DEV_PREFLIGHT_NO_FIT' or pre.get('source_sha256')!=source_hashes:
        raise ValueError('Preflight missing, stale or not provenance-bound')
    if pre.get('result',{}).get('status')!='DEV_STATISTICS_READY_ALIGNMENT_CHECK_PENDING':
        raise ValueError('DEV group/target feasibility blocked; do not fit')
    if lag.get('state')!='DEV_REFERENCE_LAG_DIAGNOSTIC_NO_FIT' or lag.get('source_sha256')!=source_hashes:
        raise ValueError('Alignment report missing or stale')
    if lag.get('status')!='LAG_SENSITIVITY_COMPUTED_NO_OFFSET_SELECTION':
        raise ValueError('Alignment sensitivity unavailable: diagnostic check needed')
    # Lag report alone is never proof of true synchronization; human review and
    # separate real user permission remain mandatory, beyond the CLI switch.
    if df.participant.nunique() < 8:
        raise ValueError("Need >=8 DEV subjects for meaningful pilot")
    groups = df.participant.to_numpy()
    x = df[list(FEATS)].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
    y = df.bad10.to_numpy(dtype=int)
    err = df.error_bpm.to_numpy(dtype=float)
    if np.unique(y).size < 2 or not np.isfinite(err).all():
        raise ValueError("Insufficient target variation or invalid error; no fit")
    models = {
        "logistic": make_pipeline(SimpleImputer(strategy="median", add_indicator=False),
                                  StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
        "hgb": make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=2, random_state=7)),
    }
    splitter = GroupKFold(n_splits=5)
    cv_scores = {name: np.full(len(df), np.nan) for name in models}
    folds = []
    oof_fold = np.full(len(df), -1, dtype=int)
    for fi, (itr, iva) in enumerate(splitter.split(x, y, groups)):
        if set(groups[itr]) & set(groups[iva]):
            raise AssertionError("Participant leakage")
        if np.unique(y[itr]).size < 2:
            raise ValueError("One-class training fold; no valid pilot")
        oof_fold[iva] = fi
        fold = {"fold":fi, "train_subjects":sorted(set(groups[itr])),
                "val_subjects":sorted(set(groups[iva])),"val_bad_rate":float(y[iva].mean())}
        for name, model in models.items():
            est = clone(model).fit(x[itr], y[itr])
            cv_scores[name][iva] = est.predict_proba(x[iva])[:,1]
        folds.append(fold)
    baselines = {
        "constant": np.zeros(len(df)),
        "motion_only": df.acc_motion.to_numpy(dtype=float),
        "spectral_peak_only": -df.ppg_concentration.to_numpy(dtype=float),
        "vendor_temporal_jump": df.vendor_hr_delta.to_numpy(dtype=float),
        "ppg_watch_disagreement": df.ppg_hr_disagree.to_numpy(dtype=float),
        "ppg_spectral_entropy": df.ppg_entropy.to_numpy(dtype=float),
        **cv_scores
    }
    metrics = {}
    for name, score in baselines.items():
        if not np.isfinite(score).all():
            raise ValueError(f"Nonfinite score {name}")
        metrics[name] = {"AUPRC": float(average_precision_score(y, score)),
                         "AUROC":float(roc_auc_score(y,score)),
                         "risk_coverage":risk_coverage(y,err,score),
                         "group_macro_risk_coverage":group_macro_risk_coverage(y,err,score,groups)}
    result = {"state":"DEV_PILOT_ONLY", "protocol":"Galaxy-HRGuard-v0.5",
              "dev_features_sha256":sha256_file(csv),
              "participants":int(df.participant.nunique()),"N_windows":len(df),
              "bad10_prevalence":float(y.mean()),"full_valid_MAE_bpm":float(err.mean()),
              "note":"All estimates are grouped OOF DEV. Individual windows overlap; no window-level significance claims.",
              "evidence_sha256":{name:sha256_file(path) for name,path in {
                    'receipt':receipt,'audit':audit,'probe':probe,'preflight':preflight,'alignment':alignment}.items()},
              "models":metrics,"folds":folds,"SEALED_subjects_UNREAD":sorted(SEAL)}
    output.parent.mkdir(parents=True,exist_ok=True)
    # Predictions permit an independent later recomputation from original DEV labels.
    pred = df[["participant","session","start_ms","end_ms","bad10","error_bpm"]].copy()
    pred["oof_fold"] = oof_fold
    for name, score in baselines.items():
        pred["risk_"+name] = score
    pred_path = output.with_name(output.stem+"_oof.csv")
    pred.to_csv(pred_path,index=False)
    result["predictions_sha256"] = sha256_file(pred_path)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","pilot"])
    p.add_argument("--windows",type=Path)
    p.add_argument("--raw",type=Path)
    p.add_argument("--csv",type=Path)
    p.add_argument("--out",type=Path)
    for evidence in ('receipt','audit','probe','preflight','alignment'):
        p.add_argument('--'+evidence,type=Path)
    p.add_argument("--allow-cpu-fit", action="store_true", help="Technical gate; requires separate actual user authorization")
    a=p.parse_args()
    if a.command == "plan":
        print(json.dumps({"dev":sorted(DEV), "sealed":sorted(SEAL),"version":SEED},indent=2))
    else:
        if not a.csv or not a.out:
            p.error("pilot requires --csv --out")
        if not a.allow_cpu_fit:
            p.error("CPU fitting is disabled until separately authorized; then pass --allow-cpu-fit")
        print(json.dumps(pilot(a.csv,a.out,receipt=a.receipt,audit=a.audit,
                               probe=a.probe,preflight=a.preflight,alignment=a.alignment),indent=2))

if __name__ == "__main__":
    main()
