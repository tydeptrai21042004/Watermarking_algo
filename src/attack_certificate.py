"""Exact bounded integer additive attack certificates for DIAL-QIM.

Applies ONLY to aligned pairwise integer pixel perturbations within [0,M].
Does not certify resampling, cropping, JPEG, continuous attacks or key security.
"""
from __future__ import annotations
import numpy as np
from .dial_qim import _view,schedule

def _cost(d,c0,c1):
    if d<0 or c0<0 or c1<0 or d>c0+c1:return (float('inf'),None)
    lo=max(0,d-c1);hi=min(d,c0)
    e0=min(hi,max(lo,d//2));e1=d-e0
    return e0*e0+e1*e1,(e0,e1)

def universal_bound(h):
    if h<2 or h%2:raise ValueError('h even and >=2 required')
    t=h//2;a=t//2;return a*a+(t-a)**2

def certify_pair(y0,y1,h,M=255):
    if h<2 or h%2:raise ValueError('h even and >=2 required')
    if not (0<=y0<=M and 0<=y1<=M):raise ValueError('invalid pixel')
    t=h//2
    specs=[('sum+',t,M-y0,M-y1,(+1,+1)),
           ('sum-',t+1,y0,y1,(-1,-1)),
           ('difference+',t,M-y0,y1,(+1,-1)),
           ('difference-',t+1,y0,M-y1,(-1,+1))]
    options=[]
    for label,delta,c0,c1,sg in specs:
        e,mods=_cost(delta,c0,c1)
        if mods is not None:
            options.append((e,label,(y0+sg[0]*mods[0],y1+sg[1]*mods[1])))
    if not options:return {'energy':float('inf'),'direction':None,'witness':None}
    energy,label,witness=min(options,key=lambda z:z[0])
    return {'energy':int(energy),'direction':label,'witness':witness}

def certify_image(marked,meta):
    """sorted independent-pair certificates + exact prefix attack energies."""
    x,_=_view(marked)
    M=int(np.iinfo(x.dtype).max)
    a,b,_=schedule(x.shape,meta.length,meta.seed)
    n=(meta.length+1)//2;K=x.shape[0]*(x.shape[1]//2)
    nc=x.shape[1]//2
    rows=[]
    for i in range(n):
        j=(a*i+b)%K;row=j//nc;col=2*(j%nc)
        cert=certify_pair(int(x[row,col,meta.channel]),int(x[row,col+1,meta.channel]),meta.h,M)
        rows.append({'index':i,'row':row,'col':col,**cert})
    rows.sort(key=lambda z:(z['energy'],z['index']))
    prefix=np.empty(n+1,dtype=np.int64);prefix[0]=0
    for k in range(n):prefix[k+1]=prefix[k]+rows[k]['energy']
    return rows,prefix


def certify_both_bits(y0, y1, h, M=255):
    """Exact least squared integer-pixel attack energy that flips BOTH bits.

    The marked pair is assumed to be on the prescribed sum/difference QIM
    centers, with the upward tie rule in `dial_qim._extract`.  The attack
    must keep both pixel values in [0, M].

    This result specializes the geometry of the two-bit lattice: any both-bit
    flip must cross a sum and a difference boundary.  Its minimum witness
    is one of four single-coordinate perturbations.
    """
    if h < 2 or h % 2:
        raise ValueError('h must be even and >=2')
    if not (0 <= y0 <= M and 0 <= y1 <= M):
        raise ValueError('invalid pixel')
    t = h // 2
    options = []
    for e0, e1, name in ((t, 0, 'sum+ diff+'),
                         (-(t+1), 0, 'sum- diff-'),
                         (0, t+1, 'sum+ diff-'),
                         (0, -(t+1), 'sum- diff+')):
        z0, z1 = y0 + e0, y1 + e1
        if 0 <= z0 <= M and 0 <= z1 <= M:
            options.append((e0*e0+e1*e1, name, (z0, z1)))
    if not options:
        return {'energy': float('inf'), 'direction': None, 'witness': None}
    energy, direction, witness = min(options, key=lambda x: x[0])
    return {'energy': int(energy), 'direction': direction, 'witness': witness}


def certify_sum_bit_only(y0, y1, h, M=255):
    """Exact cost to flip the SUM bit alone, for an odd-length last carrier."""
    if h < 2 or h % 2 or not (0 <= y0 <= M and 0 <= y1 <= M):
        raise ValueError('invalid h or pixel')
    t = h // 2
    options = []
    for label, d, c0, c1, sign in (
        ('sum+', t, M-y0, M-y1, +1),
        ('sum-', t+1, y0, y1, -1),
    ):
        energy, shifts = _cost(d, c0, c1)
        if shifts is not None:
            options.append((energy, label,
                            (y0+sign*shifts[0], y1+sign*shifts[1])))
    if not options:
        return {'energy': float('inf'), 'direction': None, 'witness': None}
    energy, direction, witness = min(options, key=lambda x: x[0])
    return {'energy': int(energy), 'direction': direction, 'witness': witness}


def exact_raw_bit_attack_curve(marked, meta, max_errors=None):
    """Sharp minimum integer-attack SSE for at least q RAW bit errors.

    Returns curve[q] for q=0,...,max_errors.  The underlying disjoint-pair
    model is exact but deliberately narrow: aligned, bounded integer-pixel
    perturbations, with the QIM metadata and no preprocessing/registration.

    Dynamic programming is O(N*max_errors) time and O(max_errors) memory
    (N=ceil(payload_bits/2)).  This is a *certification* computation and is
    not executed by the fast embedding kernel.
    """
    x, _ = _view(marked)
    M = int(np.iinfo(x.dtype).max)
    N = (meta.length+1)//2
    if max_errors is None:
        max_errors = meta.length
    max_errors = int(max_errors)
    if max_errors < 0 or max_errors > meta.length:
        raise ValueError('max_errors must lie between 0 and payload length')
    a, b, _ = schedule(x.shape, meta.length, meta.seed)
    K = x.shape[0] * (x.shape[1]//2)
    nc = x.shape[1]//2
    dp = [float('inf')] * (max_errors+1)
    dp[0] = 0
    live = 0
    for i in range(N):
        j = (a*i+b)%K
        row = j//nc
        col = 2*(j%nc)
        y0, y1 = int(x[row,col,meta.channel]), int(x[row,col+1,meta.channel])
        active = 2 if 2*i+1 < meta.length else 1
        if active == 2:
            one = certify_pair(y0,y1,meta.h,M)['energy']
            two = certify_both_bits(y0,y1,meta.h,M)['energy']
        else:
            one = certify_sum_bit_only(y0,y1,meta.h,M)['energy']
            two = float('inf')
        new = dp.copy()  # no errors from this carrier
        for q in range(min(live,max_errors)+1):
            if dp[q] == float('inf'):
                continue
            if q+1 <= max_errors:
                new[q+1] = min(new[q+1], dp[q]+one)
            if active == 2 and q+2 <= max_errors:
                new[q+2] = min(new[q+2], dp[q]+two)
        dp = new
        live += active
    return dp
