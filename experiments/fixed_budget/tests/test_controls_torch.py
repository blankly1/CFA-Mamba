"""Server CPU unit tests for adapter math. Requires torch, not CUDA or image data."""
import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'common'))
from contracts import settings

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    s=settings(os.environ.get('CFA_TEST_SETTINGS',str(ROOT/'settings.json')))
    sys.path.insert(0,str(s['code_root']/'repository'))
    from cfa_mamba.model import SemanticConditionedScanAdapter
    from variants import ConditionalMLP,ResetState,SharedCondition


@unittest.skipUnless(HAS_TORCH,'PyTorch unavailable; execute these tests in the training environment')
class ControlsTests(unittest.TestCase):
    def test_mlp_parameter_count_and_gradient(self):
        kwargs=dict(num_classes=7,feature_dim=16,hidden_dim=16)
        scan=SemanticConditionedScanAdapter(**kwargs)
        mlp=ConditionalMLP(**kwargs)
        self.assertEqual(sum(p.numel() for p in mlp.parameters()),sum(p.numel() for p in scan.parameters())+1)
        x=torch.randn(2,16,3,4,requires_grad=True)
        probs=torch.randn(2,7).softmax(1)
        output=mlp(x,probs)
        self.assertEqual(output.shape,x.shape)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_reset_removes_cross_position_propagation(self):
        module=ResetState(num_classes=7,feature_dim=16)
        x=torch.randn(2,16,12)
        decay=torch.full((2,16),0.7)
        b,c=torch.rand(2,16),torch.rand(2,16)
        expected=(b*c+module.D[None,:]).unsqueeze(-1)*x
        torch.testing.assert_close(module._scan(x,decay,b,c),expected)
        modified=x.clone();modified[:,:,0]+=1
        torch.testing.assert_close(module._scan(modified,decay,b,c)[:,:,1:],expected[:,:,1:])

    def test_shared_matches_original_when_all_conditions_equal(self):
        probs=torch.randn(1,7).softmax(1)
        scan=SemanticConditionedScanAdapter(num_classes=7,feature_dim=16).eval()
        shared=SharedCondition(probs[0],num_classes=7,feature_dim=16).eval()
        shared.load_state_dict(scan.state_dict(),strict=False)
        x=torch.randn(2,16,3,4)
        torch.testing.assert_close(scan(x,probs.expand(2,-1)),shared(x,probs.expand(2,-1)))


if __name__=='__main__':
    unittest.main()
