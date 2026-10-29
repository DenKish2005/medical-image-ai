import torch
import pytest

from research_compression.allocation import allocate_bits, actual_rate, SpatialRiskHeads
from research_compression.losses import compression_loss, bernoulli_distillation, roi_consistency
from research_compression.bridge import FeatureReplacement, make_replacement_callback


def test_allocators_and_budget_estimate():
    imp=torch.tensor([[[1.,0.,0.5],[0.3,0.8,0.2]]])
    unc=torch.tensor([[[0.2,0.1,0.7],[0.8,0.3,1.]]])
    for mode in ('uniform','random','saliency','uncertainty','combined'):
        got=allocate_bits(imp,unc,channels=4,target_bits=2000,mode=mode)
        assert got.shape==(1,2,3)
        assert set(got.flatten().tolist()).issubset({2,4,8})
        assert int(got.sum())*4 <= 2000
    impossible=allocate_bits(imp,unc,channels=40,target_bits=1)
    assert torch.all(impossible==2)
    assert actual_rate(b'hello',8)['budget_met'] is False


def test_bernoulli_kl_and_roi():
    logits=torch.randn(3,2,requires_grad=True)
    teacher=torch.randn(3,2)
    assert bernoulli_distillation(teacher,teacher).abs().item()<1e-5
    loss=bernoulli_distillation(logits,teacher)
    loss.backward()
    assert logits.grad is not None
    a=torch.zeros(2,4,2,2)
    b=torch.ones(2,4,2,2)
    mask=torch.zeros(2,2,2);mask[:,0,0]=1
    assert abs(roi_consistency(a,b,mask).item()-1)<1e-6


def test_compression_loss_backward():
    student=torch.randn(2,3,requires_grad=True)
    recon=torch.randn(2,4,2,2,requires_grad=True)
    loss,terms=compression_loss(student,torch.zeros(2,3),torch.zeros(2,3),
                                 torch.ones(2,3,dtype=torch.bool),torch.zeros_like(recon),
                                 recon,torch.tensor([400.,900.],requires_grad=True),
                                 rate_target_bits=512,roi_mask=torch.ones(2,2,2))
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and recon.grad is not None


def test_feature_bridge_hook_cleanup():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.layer=torch.nn.Conv2d(3,4,1)
        def forward(self,x):return self.layer(x)
    m=Backbone()
    src=torch.randn(1,3,2,2)
    original=m(src).detach()
    with FeatureReplacement(m,'layer',lambda f: f*.5):
        assert torch.allclose(m(src),original*.5)
    assert torch.allclose(m(src),original)
    with pytest.raises(KeyError):
        with FeatureReplacement(m,'not-present',lambda f:f):pass
