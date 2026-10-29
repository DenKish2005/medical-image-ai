import numpy as np
import pytest
import torch

from research_compression.codec import (ConditionalFeatureCodec, encode_symbols,
                                        decode_symbols, fake_quantize)


@pytest.mark.parametrize('depth', [2,4,6,8])
def test_uniform_actual_stream_roundtrip(depth):
    feature = np.linspace(-2,2,num=3*3*5,dtype=np.float32).reshape(3,3,5)
    message = encode_symbols(feature,np.full((3,5),depth),split='denseblock2')
    restored, metadata = decode_symbols(message,expected_split='denseblock2')
    assert restored.shape == feature.shape
    assert len(message)*8 > 0
    assert np.isfinite(restored).all()
    assert np.max(np.abs(feature-restored)) <= (4/(2**depth-1))+0.0001
    assert metadata['packed_nbytes'] >= 1
    assert np.all(metadata['allocation']==depth)


def test_mixed_allocation_and_corruption():
    rng=np.random.default_rng(3)
    feature=rng.normal(size=(6,2,3)).astype(np.float32)
    depths=np.array([[2,4,8],[8,6,2]],dtype=np.uint8)
    message=encode_symbols(feature,depths,split='my.layer',rate_id='10%')
    restored,header=decode_symbols(message,expected_split='my.layer')
    assert restored.shape==feature.shape
    assert np.array_equal(header['allocation'],depths)
    with pytest.raises(ValueError,match='Split layer'):
        decode_symbols(message,expected_split='other')
    with pytest.raises(ValueError):
        decode_symbols(message[:-8])
    with pytest.raises(ValueError):
        decode_symbols(b'xxxx'+message[4:])


def test_invalid_data_guard():
    feature=np.ones((3,2,2),dtype=np.float32)
    with pytest.raises(ValueError):encode_symbols(feature*np.nan,np.full((2,2),2),split='a')
    with pytest.raises(ValueError):encode_symbols(feature,np.full((2,2),3),split='a')


def test_fake_quant_ste_gradients():
    x=torch.randn(2,8,3,3,requires_grad=True)
    level=torch.full((2,3,3),4,dtype=torch.uint8)
    value=fake_quantize(x,level)
    value.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_conditional_codec_stream_and_gradients():
    codec=ConditionalFeatureCodec(6,latent_channels=3)
    x=torch.randn(2,6,2,3,requires_grad=True)
    bits=torch.full((2,2,3),4,dtype=torch.uint8)
    result=codec(x,bits,rate_fraction=.25)
    assert result.reconstructed.shape==x.shape
    assert result.estimated_bits.shape==(2,)
    (result.reconstructed.square().mean()+.00001*result.estimated_bits.mean()).backward()
    assert codec.encoder.weight.grad is not None
    codec.eval()
    msg=codec.encode(x.detach(),bits,.25,split='denseblock1')
    feature=codec.decode(msg,expected_split='denseblock1')
    assert feature.shape==x.shape
    assert all(isinstance(m,bytes) for m in msg)
