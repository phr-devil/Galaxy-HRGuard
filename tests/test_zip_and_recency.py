"""Negative controls for fixed DEV extraction and label independence."""
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "src"))
from hrguard import DEV, SEAL, FEATS
from hrguard.prepare import unpack_dev
from hrguard.windows import clean_series, window_slice
from hrguard.preflight import inspect


class DevZipTests(unittest.TestCase):
    def test_extract_only_dev_members_even_if_sealed_in_archive(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); z=base/'in.zip'; dest=base/'dev'; manifest=base/'evidence'/'manifest.json'
            dev=sorted(DEV)[0]; seal=sorted(SEAL)[0]
            with zipfile.ZipFile(z,'w') as f:
                f.writestr(f'GalaxyPPG/data/{dev}/GalaxyWatch/PPG.csv', 'timestamp,ppg,status\n1,2,0\n')
                f.writestr(f'GalaxyPPG/data/{dev}/PolarH10/HR.csv', 'phoneTimestamp,hr\n1,70\n')
                f.writestr(f'GalaxyPPG/data/{seal}/GalaxyWatch/PPG.csv', 'SEALED CONTENT MUST NOT OPEN')
            orig=zipfile.ZipFile.open
            opened=[]
            def audited_open(obj, name, *args, **kwargs):
                filename=name.filename if isinstance(name,zipfile.ZipInfo) else name
                opened.append(filename)
                self.assertNotIn(f'/{seal}/',filename)
                return orig(obj,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'open',audited_open):
                result=unpack_dev(z,dest,manifest)
            self.assertEqual(result['new_files'],2)
            self.assertEqual(len(opened),2)
            self.assertFalse((dest/seal).exists())
            self.assertTrue((dest/dev/'GalaxyWatch/PPG.csv').exists())
            self.assertEqual(json.loads(manifest.read_text())['sealed_UNOPENED'],sorted(SEAL))
            # A repeated invocation reuses DEV bytes without reopening any ZIP payload.
            with patch.object(zipfile.ZipFile,'open',audited_open):
                second=unpack_dev(z,dest,manifest)
            self.assertEqual(second['new_files'],0)
            self.assertEqual(second['reused_verified_files'],2)
            self.assertEqual(len(opened),2)

    def test_ambiguous_same_dev_member_refused(self):
        with tempfile.TemporaryDirectory() as td:
            base=Path(td);z=base/'in.zip';dev=sorted(DEV)[0]
            with zipfile.ZipFile(z,'w') as f:
                f.writestr(f'A/{dev}/GalaxyWatch/PPG.csv','1')
                f.writestr(f'B/{dev}/GalaxyWatch/PPG.csv','1')
            with self.assertRaisesRegex(ValueError,'duplicate DEV member'):
                unpack_dev(z,base/'out',base/'manifest.json')

class TimestampTests(unittest.TestCase):
    def test_duplicate_hr_rows_not_counted_as_independent_observations(self):
        df=pd.DataFrame({'timestamp':[2000,1000,1000,1000,1000],
                         'hr':[80,70,70,70,71], 'hrStatus':[1]*5})
        cleaned=clean_series(df,'timestamp',['hr','hrStatus'])
        self.assertEqual(len(cleaned),2)
        self.assertEqual(cleaned.loc[0,'hr'],71)  # stable keep-last
        self.assertLess(len(window_slice(cleaned,0,8000)),4)

    def test_stale_hr_is_reported_but_not_a_model_feature(self):
        feat={k:0.0 for k in FEATS};feat.update({'participant':sorted(DEV)[0],
              'session':'rest','start_ms':0,'end_ms':8000,
              'bad10':0,'error_bpm':5.0,'watch_hr_last_age_ms':5000.0,
              'polar_hr_last_age_ms':1000.0})
        result=inspect(pd.DataFrame([feat]))
        self.assertEqual(result['hr_recency_diagnostic']['watch_last_age_over_2s_fraction'],1)
        self.assertIn('WATCH_HR_STALENESS_GT2S_IN_MORE_THAN_10_PERCENT_WINDOWS_DIAGNOSTIC',result['warnings'])
        self.assertNotIn('watch_hr_last_age_ms',FEATS)

if __name__=='__main__':unittest.main()
