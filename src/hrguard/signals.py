"""Generate DEV features directly from RAW signals after audit + window probe.

Uses exact same fixed windows and quality conditions as probe_dev_windows.py;
interpolation only happens *after* verifying raw sampling continuity.
No sealed data, training, or model fitting.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import DEV, SEAL, signal_features, sha256_file
from .raw import FILES, read_minimal
from .windows import (WINDOW_MS, STRIDE_MS, MIN_GOOD_HR, sensor_is_usable,
                               window_slice, clean_series, sessions_from_events)

COLUMNS=['participant','session','start_ms','end_ms','watch_hr','ref_hr','error_bpm','bad10',
         'valid_hr_fraction','watch_hr_last_age_ms','polar_hr_last_age_ms',
         'vendor_hr_std','vendor_hr_delta','ppg_concentration',
         'ppg_entropy','ppg_peak_exists','ppg_hr_disagree','acc_motion','acc_ppg_overlap']


def uniform_signal(raw_df, cols, start, end, status=None):
    x=window_slice(raw_df,start,end)
    if status is not None:
        x=x[x.status==status]
    x=x.dropna(subset=list(cols))
    if x.empty: return None
    t=x.timestamp.to_numpy(dtype=float)
    vals=x[list(cols)].to_numpy(dtype=float)
    uniq, ii = np.unique(t, return_index=True)
    t=uniq; vals=vals[ii]
    if len(t)<2:return None
    # 8s x 25 Hz, no raw gap >240ms already verified in caller.
    grid=start+np.arange(200)*40
    return np.stack([np.interp(grid,t,vals[:,j]) for j in range(len(cols))],axis=1)


def extract_checked(raw, audit_file, probe_file, out, only_subjects=None):
    a=json.loads(Path(audit_file).read_text(encoding='utf-8'))
    q=json.loads(Path(probe_file).read_text(encoding='utf-8'))
    if a.get('state')!='READ_ONLY_DEV_AUDIT_NO_FIT' or q.get('state')!='DEV_LABEL_PROBE_NO_FIT':
        raise ValueError('Must run validated audit and probe before extraction')
    if q.get('audit_source_sha256')!=sha256_file(Path(audit_file)):
        raise ValueError('Audit and probe evidence do not match')
    subs=sorted(DEV) if only_subjects is None else sorted(only_subjects)
    if not set(subs).issubset(DEV):raise PermissionError('Refusing sealed/unknown subjects')
    rows=[]; logs={}
    for sid in subs:
        stat=q.get('by_subject',{}).get(sid,{})
        if stat.get('state')!='LABELS_OBSERVED_DEV_ONLY':
            logs[sid]={'state':'NOT_PROBED'}
            continue
        base=raw/sid
        sources={}
        for name,(rel,cols) in FILES.items():
            path=base/rel
            if sha256_file(path)!=a['subjects'][sid]['sha256'][name]:
                raise ValueError(f'Changed raw source since audit: {sid}:{name}')
            sources[name]=read_minimal(path,cols)
        offset=a['subjects'][sid]['time_alignment']['selected_offset_ms']
        if offset is None:raise ValueError(f'Not aligned: {sid}')
        ppg=clean_series(sources['watch_ppg'],'timestamp',['ppg','status'])
        acc=clean_series(sources['watch_acc'],'timestamp',['x','y','z'])
        wh=clean_series(sources['watch_hr'],'timestamp',['hr','hrStatus'])
        ref=clean_series(sources['polar_hr'].rename(columns={'phoneTimestamp':'timestamp'}),'timestamp',['hr'])
        ref['timestamp']+=offset
        counters={'eligible_raw':0,'feature_emitted':0,'feature_nonperiodic':0}
        for session,start,end in sessions_from_events(sources['events']):
            for t0 in range(start,end-WINDOW_MS+1,STRIDE_MS):
                t1=t0+WINDOW_MS
                r=window_slice(ref,t0,t1)
                rv=r.hr.to_numpy(dtype=float)
                good_r=r.loc[r.hr.between(40,200)]
                rv=good_r.hr.to_numpy(dtype=float)
                if len(rv)<MIN_GOOD_HR:continue
                h=window_slice(wh,t0,t1)
                good_h=h.loc[(h.hrStatus==1)&h.hr.between(40,200)]
                hv=good_h.hr.to_numpy(dtype=float)
                if len(hv)<MIN_GOOD_HR:continue
                if not sensor_is_usable(ppg,t0,t1,good_status=0):continue
                if not sensor_is_usable(acc,t0,t1):continue
                counters['eligible_raw']+=1
                x=uniform_signal(ppg,['ppg'],t0,t1,status=0)
                z=uniform_signal(acc,['x','y','z'],t0,t1)
                if x is None or z is None: raise ValueError('Probe/extract disagreed on raw waveform quality')
                w_hr=float(np.median(hv));r_hr=float(np.mean(rv))
                feat=signal_features(x,z,w_hr)
                if feat is None:raise ValueError('Unexpected feature failure after quality gate')
                counters['feature_emitted']+=1
                counters['feature_nonperiodic']+=feat['ppg_peak_exists']==0
                rows.append(dict(participant=sid,session=session,start_ms=t0,end_ms=t1,
                                 watch_hr=w_hr,ref_hr=r_hr,error_bpm=abs(w_hr-r_hr),
                                 bad10=int(abs(w_hr-r_hr)>10),valid_hr_fraction=min(1,len(hv)/8),
                                 watch_hr_last_age_ms=float(t1-good_h.timestamp.iloc[-1]),
                                 polar_hr_last_age_ms=float(t1-good_r.timestamp.iloc[-1]),
                                 vendor_hr_std=float(np.std(hv)),vendor_hr_delta=float(abs(hv[-1]-hv[0])),**feat))
        if counters['eligible_raw']!=stat['waveform_ok']:
            raise ValueError(f'Probe count does not match extractor for {sid}: {counters} vs {stat["waveform_ok"]}')
        logs[sid]=counters
    out.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows,columns=COLUMNS)
    if df.duplicated(['participant','session','start_ms','end_ms']).any():
        raise ValueError('Nonunique window key')
    # Verify labels again against the independent probe, not only total count.
    expected=sum(q['by_subject'][s].get('bad10',0) for s in subs)
    if int(df.bad10.sum())!=expected:raise ValueError('Bad10 count mismatch vs window probe')
    df.to_csv(out,index=False)
    receipt={'state':'DEV_FEATURES_EXTRACTED_NO_FIT','protocol':'Galaxy-HRGuard-v0.5',
             'audit_sha256':sha256_file(Path(audit_file)),'probe_sha256':sha256_file(Path(probe_file)),
             'dev_features_sha256':sha256_file(out),'N':len(df),'N_subjects':df.participant.nunique() if len(df) else 0,
             'bad10':int(df.bad10.sum()),'by_subject':logs,'sealed_UNREAD':sorted(SEAL)}
    out.with_name('feature_receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('raw','audit','probe','out'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    print(json.dumps(extract_checked(a.raw,a.audit,a.probe,a.out),indent=2))
if __name__=='__main__': main()
