"""Numerical parity and change-of-measure checks for fixed-Q sampling."""
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from av_weighted_restart import evaluate,length_distribution
from av_core import log_squared_heavy_tailed_weights,LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
from av_lambda_restart_mixture import everytime_lambda_mixture_upper_confidence_trajectory

class WeightedRestartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.library=Path(cls.temp.name)/'kernel.so'
        src=Path(__file__).resolve().parents[1]/'src/weighted_restart_kernel.cpp'
        subprocess.run(['clang++','-O3','-shared','-fPIC',str(src),'-o',str(cls.library)],check=True)
    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_length_split_ties_and_change_of_measure(self):
        median,long,q,r=length_distribution([1,2,2,4,8])
        self.assertEqual(median,2)
        np.testing.assert_array_equal(long,[False,True,True,True,True])
        self.assertAlmostEqual(q[0],2*q[1])
        np.testing.assert_allclose(q*r,np.full(5,.2))
        z=np.array([0,1,1,0,1]); self.assertAlmostEqual(np.dot(q,r*z),z.mean())

    def test_weighted_against_explicit_products(self):
        y=np.array([0,1.5,0,.75,0,0,0,1.5])
        lam=np.array([.1,.3,.6]); eps=np.array([.1,.4]); grid=np.unique(np.r_[np.linspace(0,1,101),eps])
        truth=np.full(len(y),.1)
        upper,covered,stops=evaluate(y,grid,lam,truth,eps,self.library,alpha=.4)
        w=log_squared_heavy_tailed_weights(len(y)); rejected=[]
        for t in range(len(y)):
            values=[]
            for u in grid:
                value=LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS-w[:t+1].sum()
                value+=sum(w[s]*np.mean(np.prod(1+lam[:,None]*(u-y[None,s:t+1]),axis=1)) for s in range(t+1))
                values.append(value)
            rej=np.array(values)>1/.4; rejected.append(rej)
            self.assertEqual(upper[t],grid[np.argmax(rej)] if rej[-1] else 1)
            self.assertEqual(covered[t],not rej[np.searchsorted(grid,truth[t])])
        for j,e in enumerate(eps):
            times=np.flatnonzero(np.array(rejected)[:,np.searchsorted(grid,e)])
            self.assertEqual(stops[j],times[0]+1 if len(times) else len(y)+1)

    def test_binary_matches_existing_lambda_mixture(self):
        z=np.r_[1,0,0,1,np.zeros(36)]
        lam=(np.arange(100)+.5)/100
        exact=everytime_lambda_mixture_upper_confidence_trajectory(z,lam,np.ones(100)/100,.05,25,'log_squared')
        grid=np.linspace(0,1,1001)
        upper,_,_=evaluate(z,grid,lam,np.zeros(len(z)),[.1],self.library)
        self.assertTrue(np.all(upper>=exact-1e-7))
        self.assertLess(np.max(upper-exact),.0010001)

    def test_scaled_state_survives_large_growth_then_contraction(self):
        # Forces scale-up and then scale-down; compare log-domain reference.
        y=np.r_[np.zeros(800),np.ones(4000)]
        lam=np.array([.9]); u=np.array([0.,.8,1.]); w=log_squared_heavy_tailed_weights(len(y))
        oracle=np.full(len(y),.8)
        _,covered,stops=evaluate(y,u,lam,oracle,[.8],self.library)
        logcapital=-np.inf; reserve=LOG_SQUARED_HEAVY_TAILED_WEIGHTS_TOTAL_MASS
        expected=[]
        for t,value in enumerate(y):
            reserve-=w[t]
            logcapital=np.logaddexp(logcapital,np.log(w[t]))+np.log1p(.9*(.8-value))
            expected.append(np.logaddexp(logcapital,np.log(reserve))<=np.log(20))
        np.testing.assert_array_equal(covered,expected)

    def test_pac_normalized_mixture_matches_reference(self):
        from av_core import normalized_log_squared_weights, binary_restart_log_e_at_threshold
        z=np.r_[1.,0.,1.,np.zeros(497)]
        epsilon=.15; zeta=epsilon/2; lam=(np.arange(100)+.5)/100/(1+zeta)
        weights=normalized_log_squared_weights(len(z))
        reference=binary_restart_log_e_at_threshold(z,epsilon-zeta,lam,np.ones(100)/100,weights)
        grid=np.unique(np.r_[0.,epsilon,1.])
        _,covered,stops=evaluate(z+zeta,grid,lam,np.full(len(z),epsilon),[epsilon],self.library,.025,weights,1.)
        rejected=reference>np.log(40)
        np.testing.assert_array_equal(covered,~rejected)
        hits=np.flatnonzero(rejected)
        self.assertEqual(stops[0],hits[0]+1 if len(hits) else len(z)+1)

    def test_invalid_factor(self):
        with self.assertRaises(ValueError):
            evaluate([1.5],[0,1],[.9],[0],[.1],self.library)

if __name__=='__main__': unittest.main()
