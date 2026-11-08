import sys,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from hrguard.raw import audit
from hrguard.windows import build_probe
from hrguard.signals import extract_checked
from hrguard import DEV,SEAL,FEATS,signal_features

class ExtractTests(unittest.TestCase):
    def test_flat_ppg_kept(self):
        f=signal_features(np.ones((200,1)),np.zeros((200,3)),75)
        self.assertEqual(f['ppg_peak_exists'],0)
        self.assertEqual(f['ppg_hr_disagree'],200)
        self.assertTrue(set(f).issubset(FEATS))

    def test_raw_checked_extraction(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);s=sorted(DEV)[0];r=root/'raw'/s
            (r/'GalaxyWatch').mkdir(parents=True);(r/'PolarH10').mkdir(parents=True)
            b=1_740_000_000_000;t=b+np.arange(500)*40
            ppg=np.sin(2*np.pi*1.25*np.arange(500)/25)
            pd.DataFrame({'timestamp':t,'ppg':ppg,'status':[0]*500}).to_csv(r/'GalaxyWatch'/'PPG.csv',index=False)
            pd.DataFrame({'timestamp':t,'x':np.sin(np.arange(500)),'y':[0]*500,'z':[0]*500}).to_csv(r/'GalaxyWatch'/'ACC.csv',index=False)
            th=b+np.arange(20)*1000
            pd.DataFrame({'timestamp':th,'hr':[85]*20,'hrStatus':[1]*20}).to_csv(r/'GalaxyWatch'/'HR.csv',index=False)
            pd.DataFrame({'phoneTimestamp':th+32400000,'hr':[70]*20}).to_csv(r/'PolarH10'/'HR.csv',index=False)
            pd.DataFrame({'timestamp':[b,b+20000],'session':['baseline']*2,'status':['ENTER','EXIT']}).to_csv(r/'Event.csv',index=False)
            aj=root/'audit.json';pj=root/'probe.json';out=root/'feat.csv'
            audit(root/'raw',aj,only_subjects={s});build_probe(root/'raw',aj,pj,only_subjects={s})
            rc=extract_checked(root/'raw',aj,pj,out,only_subjects={s})
            self.assertEqual(rc['N'],7)
            self.assertEqual(rc['bad10'],7)
            self.assertEqual(pd.read_csv(out).bad10.sum(),7)
            with self.assertRaises(PermissionError):
                extract_checked(root/'raw',aj,pj,out,only_subjects={next(iter(SEAL))})

if __name__=='__main__':unittest.main()
