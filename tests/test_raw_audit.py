import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrguard import DEV, SEAL
from hrguard.raw import infer_offset, audit

class AuditTests(unittest.TestCase):
    def test_alignment_is_only_range_based(self):
        t=1_740_000_000_000 + np.arange(100)*1000
        ref=pd.Series(t+32400000)
        watch=pd.Series(t)
        a=infer_offset(ref,watch)
        self.assertEqual(a['selected_offset_ms'],-32400000)
        self.assertEqual(a['status'],'UNIQUE_TIME_RANGE_MATCH')
        b=infer_offset(pd.Series(t+5*3600000),watch)
        self.assertIsNone(b['selected_offset_ms'])

    def test_sealed_subject_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(PermissionError):
                audit(Path(td),Path(td)/'audit.json',only_subjects={next(iter(SEAL))})

    def test_missing_real_files_is_visible_not_passed(self):
        with tempfile.TemporaryDirectory() as td:
            sid=sorted(DEV)[0]
            r=audit(Path(td),Path(td)/'out'/'audit.json',only_subjects={sid})
            self.assertEqual(r['subjects'][sid]['state'],'MISSING_REQUIRED_FILE')
            self.assertEqual(r['summary']['ready_subjects'],0)

    def test_synthetic_end_to_end_read_only_audit(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            sid=sorted(DEV)[0]
            s=root/'raw'/sid
            (s/'GalaxyWatch').mkdir(parents=True)
            (s/'PolarH10').mkdir(parents=True)
            base=1_740_000_000_000
            t=base+np.arange(500)*40
            pd.DataFrame({'timestamp':t,'ppg':np.sin(np.arange(500)/7), 'status':np.zeros(500,int)}).to_csv(s/'GalaxyWatch'/'PPG.csv',index=False)
            pd.DataFrame({'timestamp':t,'x':np.ones(500),'y':np.ones(500),'z':np.ones(500)}).to_csv(s/'GalaxyWatch'/'ACC.csv',index=False)
            hrs=base+np.arange(20)*1000
            pd.DataFrame({'timestamp':hrs,'hr':[72]*20,'hrStatus':[1]*20}).to_csv(s/'GalaxyWatch'/'HR.csv',index=False)
            pd.DataFrame({'phoneTimestamp':hrs+32400000,'hr':[72]*20}).to_csv(s/'PolarH10'/'HR.csv',index=False)
            pd.DataFrame({'timestamp':[base,base+20000],'session':['baseline','baseline'],'status':['ENTER','EXIT']}).to_csv(s/'Event.csv',index=False)
            dest=root/'report.json'
            result=audit(root/'raw',dest,only_subjects={sid})
            self.assertEqual(result['subjects'][sid]['state'],'READY_FOR_WINDOW_CHECK')
            self.assertEqual(result['subjects'][sid]['time_alignment']['selected_offset_ms'],-32400000)
            self.assertTrue(dest.is_file())
            self.assertEqual(len(json.loads(dest.read_text())['subjects']),1)

if __name__=='__main__':unittest.main()
