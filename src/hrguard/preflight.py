"""Non-fitting, provenance-aware DEV feasibility check for Galaxy-HRGuard.

Validates the audit -> window probe -> features SHA chain, raw label algebra,
participant split, group-fold feasibility and *predeclared* simple heuristics.
Does not claim model improvement or second-level timestamp synchronization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .core import DEV, SEAL, FEATS, risk_coverage, group_macro_risk_coverage, sha256_file

KEY = ['participant', 'session', 'start_ms', 'end_ms']
REQ = set(KEY) | {'watch_hr','ref_hr','error_bpm','bad10','valid_hr_fraction',
                  'watch_hr_last_age_ms','polar_hr_last_age_ms'} | set(FEATS)


def load_verified_evidence(csv: Path, receipt: Path, audit: Path, probe: Path):
    for f in (csv, receipt, audit, probe):
        if not f.is_file(): raise FileNotFoundError(f'Needed evidence absent: {f}')
    rc = json.loads(receipt.read_text(encoding='utf-8'))
    a = json.loads(audit.read_text(encoding='utf-8'))
    q = json.loads(probe.read_text(encoding='utf-8'))
    if rc.get('state')!='DEV_FEATURES_EXTRACTED_NO_FIT':
        raise ValueError('Feature receipt not an audited DEV extraction')
    if a.get('state')!='READ_ONLY_DEV_AUDIT_NO_FIT' or q.get('state')!='DEV_LABEL_PROBE_NO_FIT':
        raise ValueError('Wrong audit/probe state')
    matches = [
        (rc.get('audit_sha256'),sha256_file(audit),'receipt->audit'),
        (rc.get('probe_sha256'),sha256_file(probe),'receipt->probe'),
        (rc.get('dev_features_sha256'),sha256_file(csv),'receipt->features'),
        (q.get('audit_source_sha256'),sha256_file(audit),'probe->audit'),
    ]
    for expected,actual,label in matches:
        if expected!=actual: raise ValueError(f'Source fingerprint mismatch: {label}')
    df=pd.read_csv(csv)
    if not REQ.issubset(df.columns):
        raise ValueError(f'Missing feature columns: {sorted(REQ - set(df.columns))}')
    if len(df)!=rc.get('N') or not len(df): raise ValueError('Feature receipt count mismatch or no eligible data')
    if df[KEY].isna().any().any() or df.duplicated(KEY).any(): raise ValueError('Invalid or duplicate window keys')
    if not set(df.participant).issubset(DEV): raise PermissionError('Sealed or unknown participant in features')
    if df.participant.nunique()!=rc.get('N_subjects'): raise ValueError('Receipt participant count mismatch')
    if not df.start_ms.lt(df.end_ms).all() or not ((df.end_ms-df.start_ms)==8000).all():
        raise ValueError('Invalid 8s window geometry')
    for c in ('watch_hr','ref_hr','error_bpm','bad10','valid_hr_fraction'):
        if not np.isfinite(pd.to_numeric(df[c],errors='coerce')).all():
            raise ValueError(f'Nonfinite target/availability column: {c}')
    if not df.watch_hr.between(40,200).all() or not df.ref_hr.between(40,200).all():
        raise ValueError('Reference/watch HR outside precommitted physiological range')
    if not np.allclose(df.error_bpm.to_numpy(),np.abs(df.watch_hr-df.ref_hr),atol=1e-6,rtol=0):
        raise ValueError('Error labels inconsistent with Polar / Galaxy HR')
    if not np.array_equal(df.bad10.to_numpy(),(df.error_bpm>10).astype(int).to_numpy()):
        raise ValueError('bad10 labels not derived from >10 bpm rule')
    for age_col in ('watch_hr_last_age_ms','polar_hr_last_age_ms'):
        if not np.isfinite(df[age_col]).all() or not df[age_col].between(0,8000,inclusive='both').all():
            raise ValueError(f'Invalid or nonfinite HR recency in {age_col}')
    if not df.bad10.isin((0,1)).all() or not df.valid_hr_fraction.between(0,1).all():
        raise ValueError('Invalid bad10 or vendor HR fraction')
    if int(df.bad10.sum())!=rc.get('bad10'): raise ValueError('Bad10 count inconsistent with receipt')
    for sid in sorted(set(df.participant)):
        expected=q.get('by_subject',{}).get(sid,{});
        if expected.get('state')!='LABELS_OBSERVED_DEV_ONLY':
            raise ValueError(f'Subject absent from prior DEV label probe: {sid}')
        n=int(np.sum(df.participant==sid));p=int(df.loc[df.participant==sid,'bad10'].sum())
        if n!=expected.get('waveform_ok') or p!=expected.get('bad10'):
            raise ValueError(f'Feature/probe group mismatch {sid}')
    # This establishes a SHA chain to previous reports, not a direct re-read of raw files.
    return df, rc


def inspect(df: pd.DataFrame):
    groups=df.participant.to_numpy()
    y=df.bad10.to_numpy(dtype=int)
    err=df.error_bpm.to_numpy(dtype=float)
    stats=[]
    for sid,g in df.groupby('participant',sort=True):
        yy=g.bad10.to_numpy(dtype=int)
        stats.append({'id':sid,'n':len(g),'positive':int(yy.sum()),
                      'positive_rate':float(yy.mean()),'mae_bpm':float(g.error_bpm.mean()),
                      'n_sessions':int(g.session.nunique())})
    n_groups=len(stats)
    folds=[]
    if n_groups>=5:
        for fi,(tr,va) in enumerate(GroupKFold(n_splits=5).split(df,y,groups)):
            folds.append({'fold':fi,
                          'train_n':int(len(tr)),'val_n':int(len(va)),
                          'train_groups':int(len(set(groups[tr]))),
                          'val_groups':int(len(set(groups[va]))),
                          'train_positive_groups':int(sum(df.loc[groups==s,'bad10'].sum()>0 for s in set(groups[tr]))),
                          'val_positive_groups':int(sum(df.loc[groups==s,'bad10'].sum()>0 for s in set(groups[va]))),
                          'train_positive':int(y[tr].sum()),'val_positive':int(y[va].sum()),
                          'train_both_classes': bool(np.unique(y[tr]).size==2),
                          'val_both_classes':bool(np.unique(y[va]).size==2)})
    # Precommitted rules, no feature fitting or optimized thresholds.
    rules = {'constant':np.zeros(len(df)),
             'wrist_motion':df.acc_motion.to_numpy(dtype=float),
             'ppg_entropy':df.ppg_entropy.to_numpy(dtype=float),
             'ppg_watch_disagree':df.ppg_hr_disagree.to_numpy(dtype=float),
             'vendor_hr_jump':df.vendor_hr_delta.to_numpy(dtype=float)}
    rule_metrics={}
    if np.unique(y).size==2:
        for name,score in rules.items():
            if not np.isfinite(score).all():
                rule_metrics[name]={'status':'NONFINITE_SCORE'};continue
            rule_metrics[name]={'AUPRC':float(average_precision_score(y,score)),
                                'AUROC':float(roc_auc_score(y,score)),
                                'risk_coverage':risk_coverage(y,err,score),
                                'group_macro_risk_coverage':group_macro_risk_coverage(y,err,score,groups)}
    recency={
        'watch_last_age_median_ms':float(df.watch_hr_last_age_ms.median()),
        'watch_last_age_p95_ms':float(df.watch_hr_last_age_ms.quantile(.95)),
        'watch_last_age_over_2s_fraction':float((df.watch_hr_last_age_ms>2000).mean()),
        'polar_last_age_over_2s_fraction':float((df.polar_hr_last_age_ms>2000).mean()),
        'policy':'Diagnostic only, no post-hoc exclusion or automatic threshold tuning',
    }
    notes=[]
    if recency['watch_last_age_over_2s_fraction']>.1:
        notes.append('WATCH_HR_STALENESS_GT2S_IN_MORE_THAN_10_PERCENT_WINDOWS_DIAGNOSTIC')
    if n_groups<8: notes.append('INSUFFICIENT_SUBJECT_GROUPS_LT8')
    if len(np.unique(y))<2: notes.append('ONLY_ONE_ERROR_CLASS')
    if sum(s['positive']>0 for s in stats)<3: notes.append('ERROR_CONCENTRATED_FEWER_THAN_3_SUBJECTS')
    if folds and not all(f['train_both_classes'] for f in folds): notes.append('AT_LEAST_ONE_TRAIN_FOLD_ONE_CLASS')
    if folds and not all(f['val_both_classes'] for f in folds): notes.append('AT_LEAST_ONE_VAL_FOLD_ONE_CLASS')
    if n_groups and max(s['n'] for s in stats)/len(df)>.25: notes.append('ONE_PARTICIPANT_DOMINATES_MORE_THAN_25_PERCENT_WINDOWS')
    return {'status':'DATA_FEASIBILITY_BLOCKED' if any(z in notes for z in
                ('INSUFFICIENT_SUBJECT_GROUPS_LT8','ONLY_ONE_ERROR_CLASS','AT_LEAST_ONE_TRAIN_FOLD_ONE_CLASS'))
                else 'DEV_STATISTICS_READY_ALIGNMENT_CHECK_PENDING',
            'n_windows':len(df),'n_subjects':n_groups,'n_sessions_total':int(df.groupby(['participant','session']).ngroups),
            'positive':int(y.sum()),'bad10_rate':float(y.mean()),'global_MAE_bpm':float(np.mean(err)),
            'positive_subjects':int(sum(s['positive']>0 for s in stats)),
            'subject_macro_bad10_rate':float(np.mean([s['positive_rate'] for s in stats])),
            'subject_macro_mae_bpm':float(np.mean([s['mae_bpm'] for s in stats])),
            'by_subject':stats,'group_folds':folds,'hr_recency_diagnostic':recency,
            'untuned_rules_dev_descriptive':rule_metrics,
            'warnings':notes,
            'caution':'Exploratory DEV-only estimates; overlapping windows are not independent. No trained gain or novel Claim demonstrated.'}


def preflight(csv, receipt, audit, probe, out):
    df,_=load_verified_evidence(csv,receipt,audit,probe)
    r=inspect(df)
    report={'state':'DEV_PREFLIGHT_NO_FIT', 'result':r,
            'source_sha256':{k:sha256_file(v) for k,v in {'features':csv,'receipt':receipt,'audit':audit,'probe':probe}.items()},
            'sealed_UNREAD':sorted(SEAL),
            'gate':'No fit authorization inferred; separate local raw synchronization review is required before pilot'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for p in ('features','receipt','audit','probe','out'):ap.add_argument('--'+p,type=Path,required=True)
    args=ap.parse_args()
    r=preflight(args.features,args.receipt,args.audit,args.probe,args.out)
    print(json.dumps({'status':r['result']['status'],
                      'n_windows':r['result']['n_windows'],
                      'n_subjects':r['result']['n_subjects'],
                      'positive':r['result']['positive'],
                      'warnings':r['result']['warnings'],'report':str(args.out)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
