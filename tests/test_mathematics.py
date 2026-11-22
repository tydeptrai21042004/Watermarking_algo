"""Brute-force proof audits, not just reconstruction tests."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pytest
from src.dial_qim import embed,extract,psnr,_nearest_general
from src.attack_certificate import certify_pair,certify_image,universal_bound

def near(v,r,P,M):
    nlow=(-r+P-1)//P;nhigh=(M-r)//P
    if nlow>nhigh:return None
    n=max(nlow,min(nhigh,(2*(v-r)+P)//(2*P)))
    return r+n*P

def test_exact_lattice_projection_exhaustive():
    M=7;checked=0
    for h in (2,4):
        for ps in range(0,2*h,2):
            for pd in range(0,2*h,2):
                for bits in range(4):
                    m0=bits>>1;m1=bits&1
                    R0=(ps+pd+(m0+m1)*h)//2
                    R1=(ps-pd+(m0-m1)*h)//2
                    for x0 in range(M+1):
                        for x1 in range(M+1):
                            best=10**9
                            for branch in (0,1):
                                y0=near(x0,R0+branch*h,2*h,M)
                                y1=near(x1,R1+branch*h,2*h,M)
                                if y0 is not None and y1 is not None:
                                    best=min(best,(y0-x0)**2+(y1-x1)**2)
                            brute=min(((y0-x0)**2+(y1-x1)**2
                                      for y0 in range(M+1) for y1 in range(M+1)
                                      if (y0+y1-ps)%(2*h)==m0*h
                                      and (y0-y1-pd)%(2*h)==m1*h), default=10**9)
                            assert best==brute
                            checked+=1
    assert checked==5120

def test_exact_adversarial_attack_energy_exhaustive():
    M=7;count=0
    for h in (2,4,6):
        for y0 in range(M+1):
            for y1 in range(M+1):
                cert=certify_pair(y0,y1,h,M=M)
                ps=(y0+y1)%(2*h);pd=(y0-y1)%(2*h)
                best=min(((v0-y0)**2+(v1-y1)**2
                     for v0 in range(M+1) for v1 in range(M+1)
                     if ((v0+v1-ps+h//2)%(2*h))//h != 0
                     or ((v0-v1-pd+h//2)%(2*h))//h != 0),default=10**9)
                assert cert['energy']==best
                assert cert['energy']>=universal_bound(h)
                count+=1
    assert count==192

@pytest.mark.parametrize('shape,dtype',[
    ((63,65),np.uint8),((125,127,3),np.uint8),
    ((125,127,3),np.uint16),((251,253),np.uint16)
])
def test_variable_sizes_types_shapes(shape,dtype):
    rng=np.random.default_rng(2026)
    x=rng.integers(0,np.iinfo(dtype).max+1,shape,dtype=dtype)
    bits=rng.integers(0,2,411,dtype=np.uint8)
    y,meta=embed(x,bits)
    assert np.array_equal(extract(y,meta),bits)
    assert np.array_equal(extract(y,meta.to_json()),bits)
    assert psnr(x,y)>=55
    assert meta.length==len(bits)

def test_attack_witnesses_disjoint():
    rng=np.random.default_rng(51)
    x=rng.integers(0,256,(126,130,3),dtype=np.uint8)
    bits=rng.integers(0,2,101,dtype=np.uint8)
    y,meta=embed(x,bits)
    cert,prefix=certify_image(y,meta)
    yy=y.copy()
    for row in cert[:5]:
        yy[row['row'],row['col'],meta.channel]=row['witness'][0]
        yy[row['row'],row['col']+1,meta.channel]=row['witness'][1]
    assert np.sum((yy.astype(np.int32)-y.astype(np.int32))**2)==prefix[5]
    out=extract(yy,meta)
    count=sum(np.any(out[i:i+2]!=bits[i:i+2]) for i in range(0,len(bits),2))
    assert count>=5

def test_capacity_and_transactional_safety():
    x=np.zeros((8,8),np.uint8)
    with pytest.raises(ValueError):embed(x,np.zeros(300,np.uint8))
    x2=np.zeros((64,64),np.uint8)
    original=x2.copy()
    with pytest.raises(ValueError):embed(x2,np.ones(512,np.uint8),h=64,psnr_floor=90,inplace=True)
    assert np.array_equal(x2,original)


def test_both_bit_certificate_exhaustive():
    from src.attack_certificate import certify_both_bits
    checked = 0
    for M in (3, 5, 7, 11):
        for h in (2, 4, 6, 8, 10):
            P=h*2; t=h//2
            for y0 in range(M+1):
                for y1 in range(M+1):
                    baseline_s=(y0+y1)%P
                    baseline_d=(y0-y1)%P
                    brute=min(((z0-y0)**2+(z1-y1)**2
                        for z0 in range(M+1) for z1 in range(M+1)
                        if (((z0+z1-baseline_s+t)%P)//h)==1
                        and (((z0-z1-baseline_d+t)%P)//h)==1),
                        default=float('inf'))
                    result=certify_both_bits(y0,y1,h,M)
                    assert result['energy']==brute, (M,h,y0,y1,brute,result)
                    if result['witness'] is not None:
                        z0,z1=result['witness']
                        assert (z0-y0)**2+(z1-y1)**2 == brute
                    checked+=1
    assert checked==1300


def test_raw_bit_attack_dp_vs_brute_force():
    from src.attack_certificate import (certify_pair, certify_both_bits,
                                       certify_sum_bit_only)
    # Compare DP over two independently attacked pairs with a direct
    # enumeration of every finite-alphabet two-pair attack.
    for M,h,pairs in [
        (3,2,[(1,1),(3,3)]), (5,4,[(2,2),(5,5)]),
        (7,4,[(7,0),(3,3)]), (5,2,[(0,0),(4,5)]),
    ]:
        local=[]
        for y0,y1 in pairs:
            P=2*h; t=h//2; ps=(y0+y1)%P;pd=(y0-y1)%P
            local.append([((a-y0)**2+(b-y1)**2,
                    (((a+b-ps+t)%P)//h != 0)+
                    (((a-b-pd+t)%P)//h != 0))
                    for a in range(M+1) for b in range(M+1)])
        brute=[0]+[float('inf')]*4
        for cost0, flips0 in local[0]:
            for cost1, flips1 in local[1]:
                for q in range(1,5):
                    if flips0+flips1>=q:
                        brute[q]=min(brute[q],cost0+cost1)
        dp=[0]+[float('inf')]*4
        for y0,y1 in pairs:
            a=certify_pair(y0,y1,h,M)['energy']
            b=certify_both_bits(y0,y1,h,M)['energy']
            nxt=dp.copy()
            for q in range(5):
                if q+1<5:nxt[q+1]=min(nxt[q+1],dp[q]+a)
                if q+2<5:nxt[q+2]=min(nxt[q+2],dp[q]+b)
            dp=nxt
        assert dp==brute, (M,h,dp,brute)
