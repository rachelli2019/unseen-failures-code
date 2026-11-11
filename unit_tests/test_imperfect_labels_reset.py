"""Audit the information boundary and fixed-bet calibration formulas."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from imperfect_labels_reset_core import FalseNegativeCalibrator, OnlineNegativeResidual, label_round


class ResetTests(unittest.TestCase):
    def test_unaudited_negative_reads_neither_truth_nor_category(self):
        def forbidden(): raise AssertionError('hidden information accessed')
        seen=set()
        self.assertEqual(label_round(0,.1,.5,seen,forbidden,forbidden),(0.,0,0,0,None))
        self.assertEqual(seen,set())

    def test_false_negative_correction_and_order(self):
        calls=[]
        def audit(): calls.append('audit');return 1
        def cat(): calls.append('category');return 7
        seen=set()
        self.assertEqual(label_round(0,.1,.01,seen,audit,cat),(10.,1,1,1,1))
        self.assertEqual(calls,['audit','category']);self.assertEqual(seen,{7})

    def test_positive_predictor_precedes_audit_and_false_positives_add_nothing(self):
        calls=[]
        def audit(): calls.append('audit');return 0
        def cat(): calls.append('category');return None
        seen=set()
        self.assertEqual(label_round(1,.1,.01,seen,audit,cat),(0.,0,1,1,0))
        self.assertEqual(calls,['category','audit']);self.assertEqual(seen,set())

    def test_correction_is_unbiased_for_missed_new_category(self):
        queried=label_round(0,.2,.1,set(),lambda:1,lambda:7)[0]
        unqueried=label_round(0,.2,.9,set(),lambda:1,lambda:7)[0]
        self.assertAlmostEqual(.2*queried+.8*unqueried,1)

    def test_mixture_matches_literal_products_and_threshold_ties(self):
        c=FalseNegativeCalibrator(.05,.025,.1)
        wealth=np.ones_like(c.correction)
        rng=np.random.default_rng(8)
        for i in range(2500):
            score=float(rng.integers(101))/100
            pi=c.propensity(score)
            loss=int(rng.random()<.1) if rng.random()<pi else None
            ips=0 if loss is None else .9*loss/pi
            wealth*=1+c.kappa*(.05-ips*(score<c.thresholds[:,None]))
            c.update(score,loss,pi)
        actual=np.array([c.log_wealth(j) for j in range(len(c.thresholds))])
        np.testing.assert_allclose(actual,np.log(wealth.mean(1)),atol=1e-11)
        self.assertTrue(np.all(np.diff(actual)<=1e-10))
        self.assertAlmostEqual(c.kappa.max(),.995/(9+.05))
        c.threshold=.5
        self.assertEqual(c.propensity(.5),1)
        self.assertEqual(c.propensity(.49),.1)


    def test_native_restart_mixture_matches_explicit_products(self):
        from av_core import restart_mixture_weights_for_scheme
        from av_weighted_restart import evaluate
        y=np.array([0.,10.,0.,1.,0.,0.,0.,0.,0.,0.])
        grid=np.array([0.,.1,.2,.5,1.]);lam=np.array([.025,.075])
        weights,total=restart_mixture_weights_for_scheme('log_squared',len(y))
        expected=[];coverage=[];stop=len(y)+1
        for t in range(1,len(y)+1):
            wealth=[]
            for u in grid:
                value=total-weights[:t].sum()
                for start in range(t):
                    value+=weights[start]*np.prod(1+lam[:,None]*(u-y[None,start:t]),axis=1).mean()
                wealth.append(value)
            rejected=np.array(wealth)>1/.8
            expected.append(grid[np.flatnonzero(rejected)[0]] if rejected.any() else 1.)
            coverage.append(not rejected[1])
            if rejected[2] and stop==len(y)+1:stop=t
        upper,covered,stops=evaluate(y,grid,lam,np.full(len(y),.1),[.2],
            'artifacts/imperfect_labels_reset/weighted_restart.so',alpha=.8,
            restart_weights=weights,total_mass=total)
        np.testing.assert_allclose(upper,expected)
        np.testing.assert_array_equal(covered,coverage)
        self.assertEqual(stops[0],stop)

    def test_residual_shrinkage_and_positive_prediction(self):
        model=OnlineNegativeResidual()
        self.assertEqual(model.predict(0,.2),0)
        model.update(.2,1)
        self.assertAlmostEqual(model.predict(0,.2),1/11)
        model.update(.2,0)
        self.assertAlmostEqual(model.predict(0,.2),1/12)
        self.assertEqual(model.predict(1,.2),0)
        self.assertEqual(model.predict(0,.9),0)

    def test_residual_correction_is_signed_and_unbiased(self):
        for truth in [0,1]:
            cat=lambda: 7 if truth else None
            queried=label_round(0,.1,.01,set(),lambda:truth,cat,.25)[0]
            unqueried=label_round(0,.1,.5,set(),lambda:truth,cat,.25)[0]
            self.assertAlmostEqual(.1*queried+.9*unqueried,truth)
            if not truth:self.assertLess(queried,0)
        def forbidden():raise AssertionError('unaudited hidden information accessed')
        self.assertEqual(label_round(0,.1,.5,set(),forbidden,forbidden,.25)[0],.25)

    def test_online_residual_uses_only_past_audits_and_keeps_history(self):
        def replay(n):
            model=OnlineNegativeResidual();seen=set();predictions=[];observations=[]
            for i in range(n):
                prediction=model.predict(0,.4);predictions.append(prediction)
                # Audit every third draw; category 7 is only novel at its first audit.
                coin=.01 if i%3==0 else .9
                y,z,d,_,_=label_round(0,.1,coin,seen,lambda:1,lambda:7,prediction)
                observations.append(y)
                if d:model.update(.4,z)
            return predictions,observations,model
        short,values,model=replay(12);long,_,_=replay(30)
        self.assertEqual(short,long[:12])
        self.assertEqual(values[0],10.)
        self.assertEqual(short[0],0.)
        self.assertAlmostEqual(short[1],1/11)
        self.assertEqual(model.count.sum(),4)
        self.assertEqual(model.total.sum(),1)

if __name__=='__main__':unittest.main()
