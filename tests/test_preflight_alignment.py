import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from hrguard.raw import audit
from hrguard.windows import build_probe
from hrguard.signals import extract_checked
from hrguard.preflight import load_verified_evidence, preflight
from hrguard.alignment import sensitivity
from hrguard import DEV,sha256_file


class FeasibilityTests(unittest.TestCase):
    def fixture(self, root):
        sid=sorted(DEV)[0]
        target=root/'raw'/sid
        (target/'GalaxyWatch').mkdir(parents=True)
        (target/'PolarH10').mkdir(parents=True)
        b=1_740_000_000_000
        t=b+np.arange(750)*40
        ppg=np.sin(2*np.pi*1.25*np.arange(750)/25)
        pd.DataFrame({'timestamp':t,'ppg':ppg,'status':0}).to_csv(target/'GalaxyWatch'/'PPG.csv',index=False)
        pd.DataFrame({'timestamp':t,'x':np.sin(np.arange(750)), 'y':0,'z':0}).to_csv(target/'GalaxyWatch'/'ACC.csv',index=False)
        ht=b+np.arange(30)*1000
        pd.DataFrame({'timestamp':ht,'hr':80+np.arange(30)//10,'hrStatus':1}).to_csv(target/'GalaxyWatch'/'HR.csv',index=False)
        pd.DataFrame({'phoneTimestamp':ht+32400000,'hr':75+np.arange(30)//10}).to_csv(target/'PolarH10'/'HR.csv',index=False)
        pd.DataFrame({'timestamp':[b,b+30000], 'session':['walk']*2,
                      'status':['ENTER','EXIT']}).to_csv(target/'Event.csv',index=False)
        ev=root/'evidence';ev.mkdir()
        au=ev/'audit.json';pr=ev/'probe.json';cs=ev/'features.csv'
        audit(root/'raw',au,only_subjects={sid});build_probe(root/'raw',au,pr,only_subjects={sid})
        extract_checked(root/'raw',au,pr,cs,only_subjects={sid})
        return (root/'raw',cs,ev/'feature_receipt.json',au,pr,ev/'preflight.json',ev/'alignment.json')

    def test_provenance_preflight_and_lag(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr,cs,rc,au,pr,out,lag=self.fixture(Path(tmp))
            result=preflight(cs,rc,au,pr,out)
            self.assertEqual(result['state'],'DEV_PREFLIGHT_NO_FIT')
            self.assertEqual(result['result']['status'],'DATA_FEASIBILITY_BLOCKED') # 1 group
            q=sensitivity(rr,cs,rc,au,pr,lag)
            self.assertEqual(q['state'],'DEV_REFERENCE_LAG_DIAGNOSTIC_NO_FIT')
            self.assertGreater(q['n_all_lags_available'],0)
            self.assertEqual(q['by_lag_common_population']['0']['label_flip_fraction_vs_0'],0)

    def test_tamper_label_even_with_rehashed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr,cs,rc,au,pr,out,lag=self.fixture(Path(tmp))
            x=pd.read_csv(cs)
            x.loc[0,'bad10']=1
            x.to_csv(cs,index=False)
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                load_verified_evidence(cs,rc,au,pr)
            receipt=json.loads(rc.read_text())
            receipt['dev_features_sha256']=sha256_file(cs)
            receipt['bad10']=int(x.bad10.sum())
            rc.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'bad10 labels'):
                load_verified_evidence(cs,rc,au,pr)

    def test_raw_watch_recomputation_detects_forgery(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr,cs,rc,au,pr,out,lag=self.fixture(Path(tmp))
            x=pd.read_csv(cs)
            x.loc[0,'watch_hr']+=2
            x.loc[0,'error_bpm']=abs(x.loc[0,'watch_hr']-x.loc[0,'ref_hr'])
            x.loc[0,'bad10']=int(x.loc[0,'error_bpm']>10)
            x.to_csv(cs,index=False)
            receipt=json.loads(rc.read_text())
            receipt['dev_features_sha256']=sha256_file(cs)
            rc.write_text(json.dumps(receipt))
            # This forged CSV is internally self-consistent; raw reread catches it.
            load_verified_evidence(cs,rc,au,pr)
            with self.assertRaisesRegex(ValueError,'Raw watch HR cannot reproduce'):
                sensitivity(rr,cs,rc,au,pr,lag)

    def test_reject_incorrect_reference_and_spurious_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            rr,cs,rc,au,pr,out,lag=self.fixture(Path(tmp))
            x=pd.read_csv(cs)
            x.loc[0,'ref_hr']+=5
            x.to_csv(cs,index=False)
            receipt=json.loads(rc.read_text())
            receipt['dev_features_sha256']=sha256_file(cs)
            rc.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'Error labels inconsistent'):
                load_verified_evidence(cs,rc,au,pr)

if __name__=='__main__':unittest.main()
