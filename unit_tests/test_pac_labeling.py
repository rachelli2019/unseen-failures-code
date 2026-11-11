"""Checks for the IPW PAC algorithm, replacing tests for the superseded formula."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from av_core import (binary_restart_log_e_at_threshold, restarted_mixture_log_eprocess_at_p,
                     log_squared_heavy_tailed_weights, normalized_log_squared_weights)
from pac_labeling_core import PacCalibrator,g_bound
from unittest.mock import patch

class PacTests(unittest.TestCase):
    def test_restart_recursion_matches_explicit_products(self):
        z=np.array([1,0,0,1,0,1,0]); w=log_squared_heavy_tailed_weights(len(z))
        lam=np.array([.1,.5,.9]); mix=np.array([.2,.3,.5]); p=.1
        reference=np.logaddexp.reduce(np.array([
            restarted_mixture_log_eprocess_at_p(z,l,np.full(len(z),p),np.arange(1,len(z)+1),w,1-w.sum())+np.log(a)
            for l,a in zip(lam,mix)]),axis=0)
        actual=binary_restart_log_e_at_threshold(z,p,lam,mix,w)
        np.testing.assert_allclose(actual,reference,atol=1e-14)
    def test_normalizer(self):
        w=normalized_log_squared_weights(100000)
        self.assertTrue(.9<w.sum()<1)
        self.assertTrue(np.all(np.diff(w)<0))
    def test_queries_only_and_ipw(self):
        p=PacCalibrator([0,1,2],.1,.05)
        self.assertEqual(p.threshold(),-np.inf)
        p.update(1,0,None,.25)
        self.assertEqual(p.sum_w,0)
        p.update(1,0,1,.25)
        self.assertEqual(p.sum_w,4); self.assertEqual(p.sum_w2,16)
        np.testing.assert_array_equal(p.errors,[0,4,0])
    def test_inverse_ties_and_empty_set(self):
        p=PacCalibrator([0,1,2],.1,.05); p.n=100; p.sum_w=100; p.sum_w2=100
        p.errors[1]=10
        with patch('pac_labeling_core.g_bound',return_value=0):
            self.assertEqual(p.threshold(),1)
            # Errors at exactly 1 are NOT entrusted to the classifier.
            self.assertFalse(1<p.threshold())
        with patch('pac_labeling_core.g_bound',return_value=1):
            self.assertEqual(p.threshold(),-np.inf)
    def test_g_is_finite(self):
        for n in (2,100,3000):
            self.assertTrue(np.isfinite(g_bound(.95,.0125,n,n)))

    def test_threshold_uses_upcoming_query_probability(self):
        p=PacCalibrator([0,1],.1,.05); p.n=10; p.sum_w=10
        with patch('pac_labeling_core.g_bound',return_value=0) as g:
            p.threshold()
        self.assertAlmostEqual(g.call_args.args[0],1-.1/(2*(1-11**-.25)))

    def test_only_flagged_categories_enter_history(self):
        from run_pac_labeling import trial
        c=dict(n_draws=60,lambda_grid_points=3,epsilons=[.1,.15],alpha=.05,history='flagged_only')
        y=np.ones(20,dtype=int); h=np.zeros(20,dtype=int); u=np.zeros(20); modes=np.arange(20)
        with patch.object(PacCalibrator,'threshold',return_value=np.inf):
            _,out=trial((c,(y,h,u,modes),np.random.SeedSequence(42)))
        seen=set(); delayed=False
        missed=set()
        for i,j in enumerate(out['draws']):
            flagged=out['0.1_tilde'][i]
            expected=int(flagged and j not in seen)
            self.assertEqual(out['0.1_z'][i],expected)
            if not flagged and j not in seen: missed.add(j)
            if expected and j in missed: delayed=True
            if flagged: seen.add(j)
            self.assertAlmostEqual(out['0.1_q'][i],1-len(seen)/20)
        self.assertTrue(delayed)

if __name__=='__main__': unittest.main()
