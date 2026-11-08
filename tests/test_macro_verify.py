"""Synthetic, NO-FIT evidence test for independent grouped-metric verification."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from hrguard import FEATS, sha256_file, risk_coverage, group_macro_risk_coverage
from hrguard.macro_verify import verify


class IndependentGroupedMetricTests(unittest.TestCase):
    def test_recompute_and_detect_forged_subject_macro(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            rows=[]
            for subj in ('P02','P03','P04','P05','P06'):
                for i in range(4):
                    ref=75.0
                    watch=ref+(16 if i==3 else 2)
                    row={'participant':subj,'session':'synthetic','start_ms':i*2000,
                         'end_ms':i*2000+8000,'watch_hr':watch,'ref_hr':ref,
                         'error_bpm':abs(watch-ref),'bad10':int(abs(watch-ref)>10),
                         'watch_hr_last_age_ms':100.,'polar_hr_last_age_ms':100.,
                         'valid_hr_fraction':1.}
                    row.update({k:float(i) for k in FEATS if k != 'valid_hr_fraction'})
                    rows.append(row)
            df=pd.DataFrame(rows)
            features=p/'features.csv';df.to_csv(features,index=False)
            au=p/'audit.json';au.write_text(json.dumps({'state':'READ_ONLY_DEV_AUDIT_NO_FIT'}))
            pr=p/'probe.json';pr.write_text(json.dumps({
                'state':'DEV_LABEL_PROBE_NO_FIT','audit_source_sha256':sha256_file(au),
                'by_subject':{sid:{'state':'LABELS_OBSERVED_DEV_ONLY',
                    'waveform_ok':len(g),'bad10':int(g.bad10.sum())}
                    for sid,g in df.groupby('participant')}}))
            rc=p/'feature_receipt.json';rc.write_text(json.dumps({
                'state':'DEV_FEATURES_EXTRACTED_NO_FIT',
                'audit_sha256':sha256_file(au),'probe_sha256':sha256_file(pr),
                'dev_features_sha256':sha256_file(features),'N':len(df),
                'N_subjects':int(df.participant.nunique()),'bad10':int(df.bad10.sum())}))
            paths={'receipt':rc,'audit':au,'probe':pr}
            source={name:sha256_file(path) for name,path in {'features':features,**paths}.items()}
            pf=p/'preflight.json';pf.write_text(json.dumps({'source_sha256':source}))
            al=p/'alignment.json';al.write_text(json.dumps({'source_sha256':source}))
            paths.update({'preflight':pf,'alignment':al})
            y=df.bad10.to_numpy(dtype=int);er=df.error_bpm.to_numpy(dtype=float)
            groups=df.participant.to_numpy();score=np.zeros(len(df))
            oof=np.full(len(df),-1)
            for fi,(_,va) in enumerate(GroupKFold(5).split(df,y,groups)):oof[va]=fi
            pred=df[['participant','session','start_ms','end_ms','bad10','error_bpm']].copy()
            pred['oof_fold']=oof;pred['risk_constant']=score
            pred_file=p/'predictions.csv';pred.to_csv(pred_file,index=False)
            metric={'AUPRC':float(average_precision_score(y,score)),
                    'AUROC':float(roc_auc_score(y,score)),
                    'risk_coverage':risk_coverage(y,er,score),
                    'group_macro_risk_coverage':group_macro_risk_coverage(y,er,score,groups)}
            data={'state':'DEV_PILOT_ONLY', 'dev_features_sha256':sha256_file(features),
                  'predictions_sha256':sha256_file(pred_file),
                  'evidence_sha256':{k:sha256_file(f) for k,f in paths.items()},
                  'models':{'constant':metric}}
            result=p/'fake_result.json';result.write_text(json.dumps(data))
            checked=verify(features,pred_file,result,**paths)
            self.assertTrue(checked['verified'])
            data['models']['constant']['group_macro_risk_coverage']['keep_0.8']['macro_MAE_bpm']+=1
            result.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Subject-macro metric mismatch'):
                verify(features,pred_file,result,**paths)

if __name__=='__main__':unittest.main()
