"""DEV-only reference time-lag sensitivity diagnostic. Never optimize a lag.

Reports how >10-bpm event labels change when reference windows are probed at
predeclared +/-2s and +/-5s, while keeping watch HR and selected 0-lag labels
fixed. These comparisons are diagnostic and cannot update the official labels.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .raw import FILES,read_minimal
from .preflight import load_verified_evidence
from .windows import clean_series
from .core import SEAL,sha256_file

LAGS_MS=(-5000,-2000,0,2000,5000)


def sensitivity(raw:Path, csv:Path, receipt:Path, audit:Path, probe:Path, out:Path):
    df,_=load_verified_evidence(csv,receipt,audit,probe)
    a=json.loads(audit.read_text(encoding='utf-8'))
    stats=[]
    for sid,g in df.groupby('participant',sort=True):
        meta=a.get('subjects',{}).get(sid,{})
        ref_file=raw/sid/FILES['polar_hr'][0]
        if sha256_file(ref_file)!=meta.get('sha256',{}).get('polar_hr'):
            raise ValueError(f'Polar source changed since initial audit: {sid}')
        watch_file=raw/sid/FILES['watch_hr'][0]
        if sha256_file(watch_file)!=meta.get('sha256',{}).get('watch_hr'):
            raise ValueError(f'Watch HR source changed since initial audit: {sid}')
        watch_hr_raw=clean_series(read_minimal(watch_file,FILES['watch_hr'][1]),
                                  'timestamp',['hr','hrStatus'])
        watch_hr_raw=watch_hr_raw.loc[(watch_hr_raw.hrStatus==1)&watch_hr_raw.hr.between(40,200)]
        wt=watch_hr_raw.timestamp.to_numpy(dtype=float)
        wh=watch_hr_raw.hr.to_numpy(dtype=float)
        raw_ref=clean_series(read_minimal(ref_file,FILES['polar_hr'][1])
                             .rename(columns={'phoneTimestamp':'timestamp'}),'timestamp',['hr'])
        offset=meta.get('time_alignment',{}).get('selected_offset_ms')
        if offset not in (0,-32400000):raise ValueError(f'Invalid frozen offset: {sid}')
        raw_ref=raw_ref.loc[raw_ref.hr.between(40,200)]
        ts=raw_ref.timestamp.to_numpy(dtype=float)+offset
        vv=raw_ref.hr.to_numpy(dtype=float)
        for row in g.itertuples(index=False):
            wi=np.searchsorted(wt,row.start_ms,'left')
            wj=np.searchsorted(wt,row.end_ms,'left')
            if wj-wi<4 or abs(float(np.median(wh[wi:wj]))-row.watch_hr)>1e-6:
                raise ValueError(f'Raw watch HR cannot reproduce feature watch_hr: {sid} start={row.start_ms}')
            ref_means={}
            for lag in LAGS_MS:
                start=row.start_ms+lag;end=row.end_ms+lag
                a0=np.searchsorted(ts,start,'left');b0=np.searchsorted(ts,end,'left')
                ref_means[lag]=float(np.mean(vv[a0:b0])) if b0-a0>=4 else None
            at_zero=ref_means[0]
            if at_zero is None or abs(at_zero-row.ref_hr)>1e-6:
                raise ValueError(f'0-lag reference not reproducible: {sid} start={row.start_ms}')
            common=all(v is not None for v in ref_means.values())
            stats.append({'subject':sid,'common':common,'zero_bad10':int(row.bad10),
                          'shift_bad10':{str(l):int(abs(row.watch_hr-ref_means[l])>10)
                                         if ref_means[l] is not None else None for l in LAGS_MS},
                          'shift_error':{str(l):abs(row.watch_hr-ref_means[l])
                                         if ref_means[l] is not None else None for l in LAGS_MS}})
    common=[x for x in stats if x['common']]
    if not common:
        status='INSUFFICIENT_COMMON_WINDOWS_FOR_LAG_PROBE'
    else:
        status='LAG_SENSITIVITY_COMPUTED_NO_OFFSET_SELECTION'
    summaries={}
    for lag in LAGS_MS:
        events=np.array([z['shift_bad10'][str(lag)] for z in common],dtype=int)
        base=np.array([z['zero_bad10'] for z in common],dtype=int)
        errors=np.array([z['shift_error'][str(lag)] for z in common],dtype=float)
        summaries[str(lag)]={
            'n_common':len(common),
            'bad10_rate':float(np.mean(events)) if len(events) else None,
            'label_flip_fraction_vs_0':float(np.mean(events!=base)) if len(events) else None,
            'mean_abs_error_bpm':float(np.mean(errors)) if len(errors) else None,
        }
    report={'state':'DEV_REFERENCE_LAG_DIAGNOSTIC_NO_FIT','status':status,
            'source_sha256':{k:sha256_file(v) for k,v in {'features':csv,'receipt':receipt,'audit':audit,'probe':probe}.items()},
            'lags_ms_predeclared':list(LAGS_MS),
            'n_original_windows':len(df),'n_all_lags_available':len(common),
            'by_lag_common_population':summaries,
            'sealed_UNREAD':sorted(SEAL),
            'interpretation':'Independently rereads SHA-pinned raw HR from both devices at fixed 0-lag; additional lags are timing-fragility diagnostics only. No lag is selected using errors. Range-overlap and lag stability do not establish true second-level sync.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','features','receipt','audit','probe','out'):p.add_argument('--'+key,required=True,type=Path)
    args=p.parse_args()
    r=sensitivity(args.raw,args.features,args.receipt,args.audit,args.probe,args.out)
    print(json.dumps({'status':r['status'], 'N':r['n_original_windows'],
                      'N_common':r['n_all_lags_available'], 'results':r['by_lag_common_population'],
                      'report':str(args.out)},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
