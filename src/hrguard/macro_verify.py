"""Independent OOF recomputation: no fit, no sealed data, no external input."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

KEY = ["participant", "session", "start_ms", "end_ms"]

def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()

def verify(features, predictions, result, *, receipt=None, audit=None, probe=None,
           preflight=None, alignment=None):
    facts = json.loads(Path(result).read_text(encoding="utf-8"))
    if facts["state"] != "DEV_PILOT_ONLY" or digest(features) != facts["dev_features_sha256"]:
        raise ValueError("Feature fingerprint mismatch")
    evidence={'receipt':receipt,'audit':audit,'probe':probe,'preflight':preflight,'alignment':alignment}
    if any(path is None for path in evidence.values()):
        raise ValueError('Independent check requires full original evidence set')
    if {name:digest(path) for name,path in evidence.items()}!=facts.get('evidence_sha256'):
        raise ValueError('DEV Pilot evidence mismatch')
    from .preflight import load_verified_evidence
    load_verified_evidence(features,receipt,audit,probe)
    expected={name:digest(path) for name,path in {'features':features,'receipt':receipt,
                                                     'audit':audit,'probe':probe}.items()}
    pre=json.loads(Path(preflight).read_text(encoding='utf-8'))
    lag=json.loads(Path(alignment).read_text(encoding='utf-8'))
    if pre.get('source_sha256')!=expected or lag.get('source_sha256')!=expected:
        raise ValueError('DEV preflight/lag source link mismatch')
    if digest(predictions) != facts["predictions_sha256"]:
        raise ValueError("Prediction fingerprint mismatch")
    x = pd.read_csv(features)
    p = pd.read_csv(predictions)
    if x.duplicated(KEY).any() or p.duplicated(KEY).any():
        raise ValueError("Nonunique window key")
    if len(x) != len(p):
        raise ValueError("Prediction count mismatch")
    if not x[KEY].reset_index(drop=True).equals(p[KEY].reset_index(drop=True)):
        raise ValueError("Prediction order, keys, or group mismatch")
    if not np.allclose(x["error_bpm"],np.abs(x["watch_hr"]-x["ref_hr"]),atol=1e-6,rtol=0):
        raise ValueError("Original feature error_bpm inconsistent with watch/reference HR")
    if not np.array_equal(x["bad10"].values,(x["error_bpm"]>10).astype(int).values):
        raise ValueError("Original feature bad10 inconsistent with >10 bpm rule")
    if not np.array_equal(x["bad10"].values,p["bad10"].values):
        raise ValueError("Ground-truth labels mismatch")
    if not np.allclose(x["error_bpm"],p["error_bpm"],rtol=0,atol=1e-8):
        raise ValueError("Ground-truth errors mismatch")
    expected = np.full(len(x),-1,int)
    groups=x.participant.to_numpy()
    for fi,(_,iva) in enumerate(GroupKFold(n_splits=5).split(x, x.bad10, groups)):
        expected[iva]=fi
    if not np.array_equal(expected,p["oof_fold"].values):
        raise ValueError("OOF fold signature mismatch")
    metrics={}
    for name in facts["models"]:
        score = p["risk_"+name].to_numpy(dtype=float)
        yy=x.bad10.to_numpy(dtype=int)
        er=x.error_bpm.to_numpy(dtype=float)
        m={"AUPRC":float(average_precision_score(yy,score)),
           "AUROC":float(roc_auc_score(yy,score))}
        for key,frac in [("keep_1",1.0),("keep_0.8",0.8),("keep_0.6",0.6)]:
            n = max(1,int(np.floor(len(score)*frac)))
            threshold = np.sort(score)[n-1]
            below = score < threshold
            at = score == threshold
            count_below = int(below.sum())
            weights = below.astype(float) + at.astype(float) * ((n-count_below)/int(at.sum()))
            m[key] = {"N":n,"bad10_rate":float(np.sum(yy*weights)/n),"MAE_bpm":float(np.sum(er*weights)/n)}
            stored = facts["models"][name]["risk_coverage"][key]
            for k,v in m[key].items():
                if abs(v-stored[k])>1e-10: raise ValueError(f"Metric mismatch {name}/{key}/{k}")
        # Independently recompute equal-within-subject coverage.  The verifier
        # does NOT import the model's group-macro helper or trust stored metrics.
        stored_macro=facts['models'][name].get('group_macro_risk_coverage')
        if not isinstance(stored_macro,dict):
            raise ValueError(f'Missing subject-macro evidence {name}')
        for key,frac in [('keep_1',1.),('keep_0.8',.8),('keep_0.6',.6)]:
            vals=[]
            for sid in np.unique(groups):
                mask=groups==sid
                ys=yy[mask]; es=er[mask]; ss=score[mask]
                n=max(1,int(np.floor(len(ss)*frac)))
                t=np.partition(ss,n-1)[n-1]
                strict=ss<t
                tied=ss==t
                weights=strict.astype(float) + tied.astype(float)*((n-int(strict.sum()))/int(tied.sum()))
                vals.append((float(np.dot(weights,ys)/n),float(np.dot(weights,es)/n)))
            check={'subjects':len(vals),
                   'macro_bad10_rate':float(np.mean([q[0] for q in vals])),
                   'macro_MAE_bpm':float(np.mean([q[1] for q in vals]))}
            expected_macro=stored_macro.get(key,{})
            for field,value in check.items():
                if field not in expected_macro or abs(value-expected_macro[field])>1e-10:
                    raise ValueError(f'Subject-macro metric mismatch {name}/{key}/{field}')
        for k in ("AUPRC","AUROC"):
            if abs(m[k]-facts["models"][name][k])>1e-10:
                raise ValueError(f"Metric mismatch {name}/{k}")
        metrics[name]=m
    return {"verified":True,"state":"DEV_OOF_VERIFIED_NOT_SEALED", "N":len(x),
            "N_subjects":int(x.participant.nunique()),"source_sha256":digest(features),
            "pred_sha256":digest(predictions),"models":list(metrics)}

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--predictions",required=True,type=Path)
    a.add_argument("--results",required=True,type=Path)
    for evidence in ('receipt','audit','probe','preflight','alignment'):
        a.add_argument('--'+evidence,required=True,type=Path)
    args=a.parse_args()
    print(json.dumps(verify(args.features,args.predictions,args.results,receipt=args.receipt,
                            audit=args.audit,probe=args.probe,preflight=args.preflight,
                            alignment=args.alignment),indent=2))
