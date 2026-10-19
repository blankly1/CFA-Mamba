"""CPU-only launch-contract tests; no scientific measurements are generated."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'common'))
from contracts import METHODS, jobs, profile, preflight, write_json


class PlanningTests(unittest.TestCase):
    def test_counts_and_unique_pairs(self):
        oh=jobs('office-home',METHODS,[0,1,2],15)
        vd=jobs('VISDA-C',['output_only','pointwise_gate','scan'],[0,1,2],15)
        self.assertEqual(len(oh),72)
        self.assertEqual(len(vd),9)
        self.assertEqual(len({r['run_id'] for r in oh+vd}),81)
        self.assertEqual({r['source'] for r in oh},{'Art','Clipart','Product','RealWorld'})

    def test_wrong_method_seed_budget_rejected(self):
        for methods,seeds in [(['typo'],[0]),(['scan'],[9]),([],[0])]:
            with self.assertRaises(ValueError):jobs('office-home',methods,seeds,15)
        with self.assertRaises(ValueError):profile('office-home',14)

    def test_profile_difference_is_explicit(self):
        oh=profile('office-home')
        vd=profile('VISDA-C')
        self.assertEqual(oh['lambda_entropy'],0.1)
        self.assertEqual(vd['lambda_entropy'],0.05)
        self.assertEqual(vd['lr_factor2'],0.1)
        self.assertEqual(oh['clip_min_weight'],1.0)

    def test_missing_inputs_fail_before_training(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            write_json(d/'bundles.json',{})
            s=dict(dataset_root=d,clip_weights=d/'missing.bin',source_root=d,
                   source_bundles=d/'bundles.json',output_root=d,baseline_profile=None)
            with self.assertRaises(RuntimeError) as context:
                preflight(s,jobs('office-home',['sfmamba_original'],[0],15,[('Art','Clipart')]))
            self.assertIn('Missing source part',str(context.exception))
            self.assertIn('baseline_profile',str(context.exception))

    def test_parameter_formula(self):
        for d,c in [(768,65),(768,12),(16,7)]:
            original=8*d*d+c*d+12*d
            replacement=(3*d+1)*(d+1)+d
            old=3*d*d+5*d
            self.assertEqual(original-old+replacement,original+1)

    def test_baseline_scope_is_not_silently_extended(self):
        for dataset, epochs in [('VISDA-C',15), ('office-home',30)]:
            with self.assertRaises(ValueError):
                jobs(dataset,['sfmamba_original'],[0],epochs)

    def test_bound_profile_matches_audited_algorithm(self):
        from contracts import PACKAGE, read_json
        settings_json=read_json(PACKAGE/'settings.json')
        baseline=read_json(PACKAGE/settings_json['baseline_profile'])
        self.assertTrue(baseline['reviewed'])
        self.assertEqual(baseline['pseu_target'],'cluster')
        self.assertEqual(baseline['par_st'],1.0)
        self.assertIn('84e9f612',baseline['evidence'])


if __name__=='__main__':
    unittest.main()
