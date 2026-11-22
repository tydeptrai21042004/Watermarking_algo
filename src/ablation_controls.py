"""Controlled component removals from ONE DIAL-QIM proposal, not competing methods.

Evaluation-only variants:
  full: original, production packed exact projection
  arithmetic_projection: same codebook/solution, but direct integer arithmetic
  single_branch: remove the lower-distortion choice between parity branches
  static_phase: remove host-position phase cycling (same two-bit lattice)
  sequential_carriers: remove keyed carrier dispersion (still disjoint pairs)

All variants encode exactly TWO raw bits per pair, use the same decoder geometry,
and require the same external metadata.  They are NOT standalone proposal methods.
Only uint8 is needed for this 60-attack ablation protocol; the production proposal
also supports uint16 and other shapes.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from numba import njit
from . import dial_qim as d

VARIANTS = ('full', 'arithmetic_projection', 'single_branch',
            'static_phase', 'sequential_carriers')

@dataclass(frozen=True)
class AblationMetadata:
    method: d.Metadata
    variant: str

@njit(cache=True)
def _ablation_kernel(x,bits,a,b,h,ch,packed,phases,direct_arithmetic,first_branch_only):
    H,W,C=x.shape
    nc=W//2
    K=H*nc
    j=b%K
    aa=a%K
    sse=np.int64(0)
    clock=0
    for i in range((bits.size+1)//2):
        row=j//nc
        col=2*(j%nc)
        x0=np.int64(x[row,col,ch]);x1=np.int64(x[row,col+1,ch])
        code=(int(bits[2*i])<<1)|(int(bits[2*i+1]) if 2*i+1<bits.size else 0)
        u=np.int64(phases[clock,code,0]);v=np.int64(phases[clock,code,1])
        if direct_arithmetic:
            v00=d._nearest_general(x0,u,2*h,255)
            v01=d._nearest_general(x1,v,2*h,255)
            v10=d._nearest_general(x0,(u+h)%(2*h),2*h,255)
            v11=d._nearest_general(x1,(v+h)%(2*h),2*h,255)
        else:
            pair0=int(packed[u,x0]);pair1=int(packed[v,x1])
            v00=np.int64(pair0&255);v01=np.int64(pair1&255)
            v10=np.int64(pair0>>8);v11=np.int64(pair1>>8)
        d00=v00-x0;d01=v01-x1;d10=v10-x0;d11=v11-x1
        c0=d00*d00+d01*d01;c1=d10*d10+d11*d11
        if first_branch_only or c0<=c1:
            y0=v00;y1=v01;cost=c0
        else:
            y0=v10;y1=v11;cost=c1
        sse+=cost
        x[row,col,ch]=y0;x[row,col+1,ch]=y1
        j+=aa
        if j>=K:j-=K
        clock+=1
        if clock>=h:clock=0
    return sse

@njit(cache=True)
def _ablation_decode(x,length,seed,a,b,stride,h,ch,static_phase):
    H,W,C=x.shape;nc=W//2;K=H*nc
    j=b%K;aa=a%K;clock=0;P=2*h
    out=np.empty(length,np.uint8)
    for i in range((length+1)//2):
        row=j//nc;col=2*(j%nc)
        u=int(x[row,col,ch]);v=int(x[row,col+1,ch])
        phase_id=0 if static_phase else clock
        ps=2*((seed+stride*phase_id)%h)
        pd=2*((7*seed+(stride+2)*phase_id)%h)
        out[2*i]=((u+v-ps+h//2)%P)//h
        if 2*i+1<length:
            out[2*i+1]=((u-v-pd+h//2)%P)//h
        j+=aa
        if j>=K:j-=K
        clock+=1
        if clock>=h:clock=0
    return out

def embed_variant(image, bits, *, variant='full', h, seed=2026, psnr_floor=55.0,
                  strict=True):
    """Fixed-step ablation. Never silently retune each variant on attacked images.

    When strict=False, permit infeasible PSNR as a *diagnostic* and explicitly
    report the violation; default is strict PSNR checking.
    """
    if variant not in VARIANTS:raise ValueError(f'Unknown variant: {variant}')
    x,gray=d._view(image)
    if x.dtype!=np.uint8:raise ValueError('Ablations are evaluated on uint8 images')
    z=np.asarray(bits)
    if z.ndim==0 or z.size==0 or np.any((z!=0)&(z!=1)):
        raise ValueError('Nonempty binary bits required')
    bb=np.ascontiguousarray(z.ravel(),dtype=np.uint8)
    h=int(h)
    if h<2 or h>64 or h%2:raise ValueError('Even h from 2 through 64 required')
    a,b,stride=d.schedule(x.shape,len(bb),seed)
    if variant=='sequential_carriers':a,b=1,0
    phases=d._residue_table(h,seed,stride)
    if variant=='static_phase':
        phases=phases.copy()
        phases[:]=phases[0]
    y=x.copy()
    sse=int(_ablation_kernel(y,bb,a,b,h,1 if y.shape[2]>1 else 0,
                             d._packed_lut(h),phases,
                             variant=='arithmetic_projection',variant=='single_branch'))
    M=255
    budget=math.floor(x.size*M*M*10**(-psnr_floor/10))
    if strict and sse>budget:
        raise ValueError(f'Ablation violates PSNR floor: SSE={sse}, budget={budget}')
    info=d.Metadata(tuple(np.asarray(image).shape),len(bb),'uint8',
                    1 if y.shape[2]>1 else 0,h,int(seed),float(psnr_floor))
    return (y[:,:,0] if gray else y),AblationMetadata(info,variant)

def extract_variant(image,metadata):
    if not isinstance(metadata,AblationMetadata):
        raise TypeError('Expected AblationMetadata')
    info=metadata.method
    x,_=d._view(image)
    if tuple(np.asarray(image).shape)!=info.shape or x.dtype.name!=info.dtype:
        raise ValueError('Image shape/dtype mismatch')
    a,b,stride=d.schedule(x.shape,info.length,info.seed)
    if metadata.variant=='sequential_carriers':a,b=1,0
    return _ablation_decode(x,info.length,info.seed,a,b,stride,info.h,info.channel,
                            metadata.variant=='static_phase')
