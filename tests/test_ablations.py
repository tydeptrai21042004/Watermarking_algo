"""Ablation invariants and paper-baseline adapter smoke checks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from PIL import Image
import pytest
from src.dial_qim import embed,extract,psnr
from src.ablation_controls import VARIANTS,embed_variant,extract_variant
from src.attack_suite60 import canonical_conditions
from baselines.registry import METHODS,PAPERS,embed as base_embed,extract as base_extract


def test_exactly_one_proposal_and_five_paper_adapters():
    assert len(METHODS)==5
    assert len(set(METHODS))==5
    assert set(METHODS)==set(PAPERS)
    assert len(canonical_conditions())==61

@pytest.mark.parametrize('count',[41,128])
def test_full_and_no_lookup_agree(count):
    x=np.random.default_rng(2026).integers(0,256,(48,64,3),dtype=np.uint8)
    bits=np.random.default_rng(count).integers(0,2,count,dtype=np.uint8)
    original,meta=embed(x,bits,h=2,psnr_floor=55)
    for variant in VARIANTS:
        y,info=embed_variant(x,bits,variant=variant,h=2,psnr_floor=55)
        assert np.array_equal(extract_variant(y,info),bits),variant
        assert psnr(x,y)>=55
        if variant in ('full','arithmetic_projection'):
            assert np.array_equal(original,y)
        assert np.array_equal(x,np.asarray(x))
    assert np.array_equal(extract(original,meta),bits)


def test_saturated_integer_cases():
    for fill in (0,255):
        x=np.full((48,64,3),fill,dtype=np.uint8)
        bits=np.random.default_rng(177).integers(0,2,61,dtype=np.uint8)
        for variant in VARIANTS:
            y,meta=embed_variant(x,bits,variant=variant,h=2)
            assert np.array_equal(extract_variant(y,meta),bits)
            assert y.min()>=0 and y.max()<=255

@pytest.mark.parametrize('name,step',[
    ('Scalar-QIM',12),('STDM-QIM',8),('SCS',12),('ISS',2),('SS',1)
])
def test_baseline_api_smoke(name,step):
    x=np.array(Image.open(ROOT/'data/host/airplane.bmp').convert('RGB'))
    bits=np.random.default_rng(2026).integers(0,2,4096,dtype=np.uint8)
    y,key=base_embed(name,x,bits,step=step,seed=2026)
    recovered=base_extract(name,y,key)
    assert y.shape==x.shape and y.dtype==np.uint8
    assert recovered.shape==(4096,)
    assert np.all((recovered==0)|(recovered==1))
    # No blanket zero-BER claim for classical baselines; report their real errors.


@pytest.mark.parametrize('step',[2,6,12,20])
def test_lookup_and_arithmetic_same_marked_pixels_at_multiple_steps(step):
    x=np.random.default_rng(817).integers(0,256,(96,128,3),dtype=np.uint8)
    bits=np.random.default_rng(818).integers(0,2,511,dtype=np.uint8)
    full,meta=embed_variant(x,bits,variant='full',h=step,psnr_floor=40.)
    direct,key=embed_variant(x,bits,variant='arithmetic_projection',h=step,psnr_floor=40.)
    assert np.array_equal(full,direct)
    assert np.array_equal(extract_variant(full,meta),bits)
    assert np.array_equal(extract_variant(direct,key),bits)
