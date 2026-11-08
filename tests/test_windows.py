import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrguard import DEV,SEAL
from hrguard.raw import audit
from hrguard.windows import build_probe,sessions_from_events,sensor_is_usable

class WindowTests(unittest.TestCase):
    def test_session_pairing(self):
        e=pd.DataFrame({'timestamp':[0,12000,12500,22500],
                        'session':['baseline','baseline','walk','walk'],
                        'status':['ENTER','EXIT','ENTER','EXIT']})
        self.assertEqual(len(sessions_from_events(e)),2)

    def test_gap_rejected_not_interpolated(self):
        t=np.arange(0,8000,40)
        df=pd.DataFrame({'timestamp':t,'ppg':np.sin(t),'status':np.zeros(len(t),int)})
        self.assertTrue(sensor_is_usable(df,0,8000,good_status=0))
        dg=df[~df.timestamp.between(2100,3000)]
        self.assertFalse(sensor_is_usable(dg,0,8000,good_status=0))

    def test_synthetic_dev_and_sealed_gate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);sid=sorted(DEV)[0];s=root/'raw'/sid
            (s/'GalaxyWatch').mkdir(parents=True)
            (s/'PolarH10').mkdir(parents=True)
            t0=1_740_000_000_000
            tm=t0+np.arange(500)*40
            pd.DataFrame({'timestamp':tm,'ppg':np.sin(np.arange(500)), 'status':[0]*500}).to_csv(s/'GalaxyWatch'/'PPG.csv',index=False)
            pd.DataFrame({'timestamp':tm,'x':[1]*500,'y':[2]*500,'z':[3]*500}).to_csv(s/'GalaxyWatch'/'ACC.csv',index=False)
            h=t0+np.arange(20)*1000
            pd.DataFrame({'timestamp':h,'hr':[80]*20,'hrStatus':[1]*20}).to_csv(s/'GalaxyWatch'/'HR.csv',index=False)
            pd.DataFrame({'phoneTimestamp':h+32400000,'hr':[70]*20}).to_csv(s/'PolarH10'/'HR.csv',index=False)
            pd.DataFrame({'timestamp':[t0,t0+20000],'session':['baseline']*2,'status':['ENTER','EXIT']}).to_csv(s/'Event.csv',index=False)
            aj=root/'audit.json';pj=root/'probe.json'
            audit(root/'raw',aj,only_subjects={sid})
            r=build_probe(root/'raw',aj,pj,only_subjects={sid})
            self.assertEqual(r['summary']['waveform_ok'],7)
            self.assertEqual(r['summary']['bad10'],0) # exact threshold is >10, not >=10
            with self.assertRaises(PermissionError):
                build_probe(root/'raw',aj,pj,only_subjects={next(iter(SEAL))})

if __name__=='__main__':unittest.main()

class MissingWaveformTests(unittest.TestCase):
    def test_nan_and_large_holes_rejected(self):
        t=np.arange(0,8000,40)
        df=pd.DataFrame({'timestamp':t,'ppg':np.sin(t), 'status':0})
        df.loc[df.timestamp.between(1000,2800),'ppg']=np.nan
        self.assertFalse(sensor_is_usable(df,0,8000,good_status=0))
