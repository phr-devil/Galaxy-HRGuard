import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hrguard import DEV, SEAL, SUBJECTS, signal_features, risk_coverage, group_macro_risk_coverage, pilot


class TestProtocol(unittest.TestCase):
    def test_holdout_identity(self):
        self.assertEqual(len(DEV), 18)
        self.assertEqual(len(SEAL), 6)
        self.assertEqual(DEV | SEAL, set(SUBJECTS))
        self.assertFalse(DEV & SEAL)

    def test_signal_feature_shapes(self):
        t = np.arange(200) / 25
        p = np.sin(2*np.pi*1.25*t)
        a = np.column_stack([np.sin(2*np.pi*1.5*t), np.zeros(200), np.zeros(200)])
        z = signal_features(p.reshape(-1, 1), a, 75)
        self.assertIsNotNone(z)
        self.assertLess(z['ppg_hr_disagree'], 5)
        self.assertLess(z['ppg_entropy'], 1)
        self.assertIsNone(signal_features(p[:100].reshape(-1,1), a[:100], 75))

    def test_score_order(self):
        d = risk_coverage(np.array([0,1,0,1]),np.array([1,14,2,13]),np.array([0,9,1,8]))
        self.assertEqual(d['keep_0.6']['bad10_rate'],0)


    def test_constant_risk_has_no_order_advantage(self):
        y=np.array([0,0,1,1,1])
        errors=np.array([1.,2.,11.,14.,20.])
        m=risk_coverage(y,errors,np.zeros(5))
        self.assertAlmostEqual(m['keep_0.8']['MAE_bpm'],np.mean(errors))
        self.assertAlmostEqual(m['keep_0.6']['bad10_rate'],np.mean(y))
    def test_group_macro_prevents_volume_domination(self):
        # A large, easy participant dominates the pooled result; the smaller
        # difficult participant still counts 50% in the per-person macro mean.
        y=np.array([0]*100+[1]*10)
        errors=np.array([2.]*100+[20.]*10)
        risk=np.array([0.]*100+[10.]*10)
        groups=np.array(['P02']*100+['P03']*10)
        pooled=risk_coverage(y,errors,risk)
        subject=group_macro_risk_coverage(y,errors,risk,groups)
        self.assertEqual(pooled['keep_0.8']['bad10_rate'],0)
        self.assertAlmostEqual(subject['keep_0.8']['macro_bad10_rate'],0.5)
        self.assertAlmostEqual(subject['keep_0.8']['macro_MAE_bpm'],11)
        self.assertEqual(subject['keep_0.8']['subjects'],2)

    def test_group_macro_is_permutation_and_tie_invariant(self):
        y=np.array([0,1,0,1,1,0])
        e=np.array([1,20,2,11,14,3.])
        g=np.array(['A','A','A','B','B','B'])
        s=np.zeros(len(y))
        one=group_macro_risk_coverage(y,e,s,g)
        rng=np.random.default_rng(17)
        for _ in range(5):
            ix=rng.permutation(len(y))
            two=group_macro_risk_coverage(y[ix],e[ix],s[ix],g[ix])
            self.assertEqual(one,two)
        for f in ('keep_0.8','keep_0.6'):
            self.assertAlmostEqual(one[f]['macro_MAE_bpm'],np.mean([np.mean(e[:3]),np.mean(e[3:])]))

    def test_pilot_refuses_sealed(self):
        with tempfile.TemporaryDirectory() as t:
            f = Path(t)/'leaked.csv'
            pd.DataFrame({'participant':[next(iter(SEAL))], 'bad10':[0], 'error_bpm':[2.0],
                          **{k:[0.1] for k in __import__('hrguard').FEATS}}).to_csv(f,index=False)
            with self.assertRaises(PermissionError):
                pilot(f,Path(t)/'result.json')


if __name__ == '__main__':
    unittest.main()
