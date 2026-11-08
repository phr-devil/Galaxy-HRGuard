"""DEV-only RAW window/label diagnostics. No interpolation, fitting, or sealed access.

Produces a scientific feasibility/label audit, not publishable performance.
Only subjects passing prior timestamp RANGE audit are processed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import DEV, SEAL, sha256_file
from .raw import FILES, read_minimal

WINDOW_MS, STRIDE_MS = 8000, 2000
MIN_SAMPLE_FRAC = 0.8  # must be 80% of 25 Hz nominal count
MAX_GAP_MS = 240
END_TOLERANCE_MS = 240
MIN_GOOD_HR = 4


def clean_series(df, timecol, signals):
    out=df.copy()
    for c in (timecol,*signals): out[c]=pd.to_numeric(out[c],errors='coerce')
    # Preserve the last measurement for duplicated source timestamps.
    # Never let duplicated HR records masquerade as independent observations.
    out=(out.dropna(subset=[timecol])
             .sort_values(timecol, kind='mergesort')
             .drop_duplicates(subset=[timecol], keep='last'))
    return out.reset_index(drop=True)


def sessions_from_events(df):
    rows=[]
    d=df.copy(); d['timestamp']=pd.to_numeric(d['timestamp'], errors='coerce')
    d=d.dropna(subset=['timestamp']).sort_values('timestamp')
    for _,g in d.groupby('session',sort=False):
        entering=[]
        for _,v in g.iterrows():
            kind=str(v.status).upper()
            if kind=='ENTER': entering.append(int(v.timestamp))
            elif kind=='EXIT' and entering:
                start=entering.pop(0)
                end=int(v.timestamp)
                if end-start>=WINDOW_MS: rows.append((str(v.session),start,end))
    return sorted(rows,key=lambda x:x[1])


def window_slice(df, start, end):
    t=df['timestamp'].to_numpy(dtype=float)
    return df.iloc[np.searchsorted(t,start,side='left'):np.searchsorted(t,end,side='left')]


def sensor_is_usable(df, start, end, good_status=None):
    x=window_slice(df,start,end)
    if good_status is not None:
        x=x[x.status==good_status]
    # Exclude missing/nonfinite values rather than counting unusable samples.
    channels=['ppg'] if 'ppg' in x else [v for v in ('x','y','z') if v in x]
    x=x.dropna(subset=channels).drop_duplicates('timestamp')
    if len(x) and (not np.isfinite(x[channels].to_numpy(dtype=float)).all()):
        x=x.loc[np.isfinite(x[channels].to_numpy(dtype=float)).all(axis=1)]
    t=x.timestamp.to_numpy(dtype=float)
    if len(t)<(25*8*MIN_SAMPLE_FRAC):return False
    if t[0]-start>END_TOLERANCE_MS or end-t[-1]>END_TOLERANCE_MS:return False
    if len(t)>1 and np.max(np.diff(t))>MAX_GAP_MS:return False
    return True


def build_probe(raw, audit_json, out, only_subjects=None):
    facts=json.loads(Path(audit_json).read_text(encoding='utf-8'))
    if facts.get('state')!='READ_ONLY_DEV_AUDIT_NO_FIT':raise ValueError('Need validated raw audit')
    subjects=sorted(DEV) if only_subjects is None else sorted(only_subjects)
    if not set(subjects).issubset(DEV):raise PermissionError('Refusing SEALED/unknown subjects')
    report={'state':'DEV_LABEL_PROBE_NO_FIT', 'protocol':'Galaxy-HRGuard-v0.5',
            'audit_source_sha256':sha256_file(Path(audit_json)),
            'sealed_subjects_UNREAD':sorted(SEAL),'by_subject':{}}
    totals={k:0 for k in ('raw_windows','reference_ok','watch_ok','waveform_ok','bad10')}
    for sid in subjects:
        a=facts.get('subjects',{}).get(sid,{})
        if a.get('state')!='READY_FOR_WINDOW_CHECK':
            report['by_subject'][sid]={'state':'SKIPPED_NOT_AUDIT_READY','audit_state':a.get('state')}
            continue
        sources={}
        for name,(rel,cols) in FILES.items():
            source=raw/sid/rel
            if sha256_file(source)!=a['sha256'].get(name):
                raise ValueError(f'Original raw file changed after audit: {sid} {name}')
            sources[name]=read_minimal(source,cols)
        offset=a['time_alignment']['selected_offset_ms']
        if offset is None: raise ValueError('Missing verified timestamp candidate')
        ppg=clean_series(sources['watch_ppg'],'timestamp',['ppg','status'])
        acc=clean_series(sources['watch_acc'],'timestamp',['x','y','z'])
        wh=clean_series(sources['watch_hr'],'timestamp',['hr','hrStatus'])
        ref=clean_series(sources['polar_hr'].rename(columns={'phoneTimestamp':'timestamp'}),'timestamp',['hr'])
        ref['timestamp']+=offset
        events=sessions_from_events(sources['events'])
        counters={k:0 for k in totals}
        per_session={}
        errors=[]
        for session,start,end in events:
            c=per_session.setdefault(session,{k:0 for k in totals})
            for t0 in range(start,end-WINDOW_MS+1,STRIDE_MS):
                t1=t0+WINDOW_MS
                c['raw_windows']+=1
                counters['raw_windows']+=1
                r=window_slice(ref,t0,t1)
                rv=r.hr.to_numpy(dtype=float)
                rv=rv[np.isfinite(rv) & (rv>=40) & (rv<=200)]
                if len(rv)<MIN_GOOD_HR: continue
                for d in (c,counters):d['reference_ok']+=1
                h=window_slice(wh,t0,t1)
                hv=h.loc[(h.hrStatus==1)&h.hr.between(40,200),'hr'].to_numpy(dtype=float)
                if len(hv)<MIN_GOOD_HR:continue
                for d in (c,counters):d['watch_ok']+=1
                if not sensor_is_usable(ppg,t0,t1,good_status=0):continue
                if not sensor_is_usable(acc,t0,t1):continue
                for d in (c,counters):d['waveform_ok']+=1
                err=float(abs(np.median(hv)-np.mean(rv)))
                errors.append(err)
                if err>10:
                    for d in (c,counters):d['bad10']+=1
        report['by_subject'][sid]={'state':'LABELS_OBSERVED_DEV_ONLY',**counters,
                                    'bad10_rate':counters['bad10']/counters['waveform_ok'] if counters['waveform_ok'] else None,
                                    'mean_abs_error_bpm':float(np.mean(errors)) if errors else None,
                                    'sessions':per_session,'offset_recipe':a['time_alignment']['selected_recipe']}
        for k in totals:totals[k]+=counters[k]
    n_mixed=sum(1 for v in report['by_subject'].values() if v.get('waveform_ok',0)>0 and v.get('bad10',0)>0 and v['bad10']<v['waveform_ok'])
    n_pos=sum(1 for v in report['by_subject'].values() if v.get('bad10',0)>0)
    report['summary']={**totals,'subject_with_bad10':n_pos,'subject_with_both_classes':n_mixed,
        'bad10_rate':totals['bad10']/totals['waveform_ok'] if totals['waveform_ok'] else None,
        'eligibility_after_reference':totals['waveform_ok']/totals['reference_ok'] if totals['reference_ok'] else None,
        'recommended_state':'CHECK_SYNCHRONIZATION_AND_GROUP_SPLIT' if totals['waveform_ok'] else 'BLOCKED_NO_VALID_WINDOWS',
        'limitation':'Time RANGE agreement is not second-level physiological synchronization; labels remain DEV exploratory.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',required=True,type=Path)
    p.add_argument('--audit',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    r=build_probe(a.raw,a.audit,a.out)
    print(json.dumps({'state':r['state'],'summary':r['summary'],'report':str(a.out)},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
