"""Read-only, DEV-only GalaxyPPG data audit; no models, labels, or sealed reads.

The official authors' processor subtracts nine hours from Polar phoneTimestamp;
this audit compares that proposed adjustment with no adjustment using *only*
timestamp overlap. It never selects an offset using heart-rate error.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import DEV, SEAL, sha256_file

FILES = {
    'watch_ppg': ('GalaxyWatch/PPG.csv', ('timestamp', 'ppg', 'status')),
    'watch_acc': ('GalaxyWatch/ACC.csv', ('timestamp', 'x', 'y', 'z')),
    'watch_hr': ('GalaxyWatch/HR.csv', ('timestamp', 'hr', 'hrStatus')),
    'polar_hr': ('PolarH10/HR.csv', ('phoneTimestamp', 'hr')),
    'events': ('Event.csv', ('timestamp', 'session', 'status')),
}
# Predeclared by the official Step02 processor; not chosen from error labels.
POLAR_CANDIDATE_OFFSETS_MS = {'raw': 0, 'author_minus_9h': -32400000}


def read_minimal(path: Path, cols):
    """Fail clearly on unexpected schema, without reading other modalities."""
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(cols) - set(columns))
    if missing:
        raise ValueError(f'{path}: required columns absent: {missing}; got {columns}')
    return pd.read_csv(path, usecols=list(cols), low_memory=False)


def numeric(col):
    return np.asarray(pd.to_numeric(col, errors='coerce'), dtype=float)


def stream_stats(df, timecol, nominal_hz=None, status=None, good=None):
    t = numeric(df[timecol]); good_t = t[np.isfinite(t)]; good_t.sort()
    out = {'records': int(len(df)), 'valid_timestamps': int(len(good_t)),
           'timestamp_min_ms': float(good_t[0]) if len(good_t) else None,
           'timestamp_max_ms': float(good_t[-1]) if len(good_t) else None,
           'duplicate_timestamps': int(len(good_t)-len(np.unique(good_t)))}
    dt = np.diff(good_t)
    pos = dt[dt>0]
    out['median_gap_ms'] = float(np.median(pos)) if len(pos) else None
    out['p95_gap_ms'] = float(np.quantile(pos,.95)) if len(pos) else None
    out['max_gap_ms'] = float(np.max(pos)) if len(pos) else None
    out['effective_hz_from_median_gap'] = float(1000 / np.median(pos)) if len(pos) else None
    if nominal_hz:
        out['nominal_hz'] = nominal_hz
    if status:
        s = numeric(df[status]); finite = np.isfinite(s)
        out['status_counts'] = {str(int(k) if float(k).is_integer() else float(k)): int(v)
                                for k,v in zip(*np.unique(s[finite],return_counts=True))}
        out['valid_status_fraction'] = float(np.mean(s==good)) if len(s) else None
    return out


def overlap_fraction(test, reference):
    """fraction of Polar measurements inside watch timestamps, no HR values."""
    a = numeric(test); b = numeric(reference)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if not len(a) or not len(b): return 0.0
    lo,hi = np.min(b),np.max(b)
    return float(np.mean((a >= lo) & (a <= hi)))


def infer_offset(reference_times, watch_times):
    """Only unique, decisive time-range agreement is accepted."""
    r = numeric(reference_times); w = numeric(watch_times)
    score = {k: overlap_fraction(r + v, w) for k,v in POLAR_CANDIDATE_OFFSETS_MS.items()}
    ranks = sorted(score, key=score.get, reverse=True)
    best, second = ranks[:2]
    winner = best if score[best] >= .75 and (score[best]-score[second]) >= .30 else None
    return {'candidate_overlap_fraction':score,
            'selected_offset_ms': POLAR_CANDIDATE_OFFSETS_MS[winner] if winner else None,
            'selected_recipe':winner, 'status': 'UNIQUE_TIME_RANGE_MATCH' if winner else 'AMBIGUOUS_OR_NO_MATCH'}


def event_stats(df, watch_times):
    e = df.copy(); e['timestamp'] = pd.to_numeric(e['timestamp'],errors='coerce')
    e = e.dropna(subset=['timestamp'])
    en = e[e.status.astype(str).str.upper()=='ENTER']
    ex = e[e.status.astype(str).str.upper()=='EXIT']
    w = numeric(watch_times); w = w[np.isfinite(w)]
    within = float(np.mean((en.timestamp >= np.min(w)) & (en.timestamp <= np.max(w)))) if len(en) and len(w) else None
    return {'rows':len(e),'enter':len(en),'exit':len(ex),'event_enter_inside_watch_range':within}


def audit(raw: Path, output: Path, only_subjects=None):
    # only_subjects exists for synthetic tests. Production CLI always uses all DEV.
    subjects = sorted(DEV) if only_subjects is None else sorted(only_subjects)
    if not set(subjects).issubset(DEV):
        raise PermissionError('Sealed/unknown subject requested: refusing before data access')
    report = {'protocol':'Galaxy-HRGuard-v0.5-raw-audit',
              'state':'READ_ONLY_DEV_AUDIT_NO_FIT',
              'dev_subjects':sorted(DEV), 'sealed_subjects_UNREAD':sorted(SEAL),
              'subjects':{}, 'summary':{}}
    count_ready = 0
    for sid in subjects:
        root = raw / sid
        missing = [rel for rel,_ in FILES.values() if not (root/rel).is_file()]
        item = {'missing_files':missing, 'sha256':{}, 'streams':{}}
        report['subjects'][sid] = item
        if missing:
            item['state']='MISSING_REQUIRED_FILE'
            continue
        try:
            data = {}
            for name,(rel,cols) in FILES.items():
                path = root/rel
                data[name] = read_minimal(path, cols)
                item['sha256'][name] = sha256_file(path)
            item['streams']['watch_ppg']=stream_stats(data['watch_ppg'],'timestamp',25,'status',0)
            item['streams']['watch_acc']=stream_stats(data['watch_acc'],'timestamp',25)
            item['streams']['watch_hr']=stream_stats(data['watch_hr'],'timestamp',1,'hrStatus',1)
            item['streams']['polar_hr']=stream_stats(data['polar_hr'],'phoneTimestamp',1)
            item['events'] = event_stats(data['events'], data['watch_ppg']['timestamp'])
            item['time_alignment'] = infer_offset(data['polar_hr']['phoneTimestamp'], data['watch_ppg']['timestamp'])
            # No labels or error thresholds are consulted here.
            item['state'] = 'READY_FOR_WINDOW_CHECK' if (
                item['time_alignment']['status']=='UNIQUE_TIME_RANGE_MATCH'
                and len(data['watch_ppg']) and len(data['watch_acc'])
                and np.sum(numeric(data['watch_hr']['hrStatus'])==1)>0
                and np.sum((numeric(data['polar_hr']['hr'])>=40) & (numeric(data['polar_hr']['hr'])<=200))>0
                and item['events']['enter']>0) else 'BLOCKED_ALIGNMENT_OR_AVAILABILITY'
            count_ready += item['state']=='READY_FOR_WINDOW_CHECK'
        except (ValueError, TypeError, KeyError, pd.errors.ParserError) as e:
            item['state']='SCHEMA_OR_PARSE_ERROR'
            item['error']=str(e)
    report['summary']={'subjects_audited':len(subjects), 'ready_subjects':count_ready,
                       'state_counts':{k:sum(x['state']==k for x in report['subjects'].values())
                                       for k in sorted(set(x['state'] for x in report['subjects'].values()))},
                       'next':'Run raw checked 8s window audit; no model fitting permitted by this audit'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True,type=Path,help='folder that contains P01,...,P24')
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    r=audit(a.raw,a.out)
    print(json.dumps({'state':r['state'],'summary':r['summary'],'report':str(a.out)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
