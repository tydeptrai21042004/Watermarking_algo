"""DIAL-QIM: Dual-moment Integer Algebraic Lattice watermarking.

ONE proposed codebook. Two arithmetically exact implementations:
* uint8: a packed finite-residue lookup kernel (fast path)
* uint16: direct exact 2-branch projection (no large lookup tables)

Variable image geometry and message length; zero clean BER upon successful
embedding, aligned blind decoder, and exactly checked global SSE/PSNR.

NOT cryptographically secure: affine carrier selection and periodic phases are
predictable. No ECC, repetition, denoising, attack-dependent embedding,
attack classifier, DCT/FFT, or geometric registration.
"""
from __future__ import annotations
from functools import lru_cache
from dataclasses import dataclass,asdict
import json,math
import numpy as np
from numba import njit

@dataclass(frozen=True)
class Metadata:
    shape: tuple[int, ...]
    length: int
    dtype: str
    channel: int
    h: int
    seed: int
    psnr_floor: float
    def to_json(self):return json.dumps(asdict(self),indent=2)
    @classmethod
    def from_json(cls,t):
        p=json.loads(t);p['shape']=tuple(p['shape']);return cls(**p)

def _view(image):
    x=np.asarray(image)
    if x.dtype not in (np.uint8,np.uint16):raise ValueError('uint8 or uint16 required')
    if x.ndim==2 and all(a>0 for a in x.shape):return np.ascontiguousarray(x[:,:,None]), True
    if x.ndim==3 and x.shape[2] in (1,3,4) and all(a>0 for a in x.shape):
        return np.ascontiguousarray(x),False
    raise ValueError('Expected nonempty HxW or HxWx(1/3/4)')

def schedule(shape,length,seed):
    H,W=shape[:2];K=H*(W//2);N=(length+1)//2
    if N>K:raise ValueError('Payload exceeds distinct two-pixel carriers')
    if N==0:raise ValueError('No payload bits')
    if not isinstance(seed,(int,np.integer)) or seed<0 or seed>=1<<63:
        raise ValueError('Nonnegative seed <2^63 required')
    a=2*(int(seed)%65536)+1
    while math.gcd(a,K)!=1:a+=2
    b=(int(seed)*23+59)%K
    return a,b,17

@lru_cache(maxsize=128)
def _packed_lut(h):
    """Packed projections on both parity cosets, O(256h) tiny fixed table."""
    P=2*h
    out=np.empty((P,256),dtype=np.uint16)
    for r in range(P):
        alt=(r+h)%P
        for v in range(256):
            n0=max(0,min((255-r)//P,(2*(v-r)+P)//(2*P)))
            n1=max(0,min((255-alt)//P,(2*(v-alt)+P)//(2*P)))
            out[r,v]=(r+P*n0)|((alt+P*n1)<<8)
    return out

@lru_cache(maxsize=128)
def _residue_table(h,seed,stride):
    # Both phases even, making (sum +/- difference)/2 integral.
    table=np.empty((h,4,2),dtype=np.uint8)
    for i in range(h):
        ps=2*((seed+stride*i)%h)
        pd=2*((7*seed+(stride+2)*i)%h)
        for code in range(4):
            b0=code>>1;b1=code&1
            u=ps+b0*h;v=pd+b1*h
            table[i,code,0]=((u+v)//2)%(2*h)
            table[i,code,1]=((u-v)//2)%(2*h)
    return table

@njit(cache=True)
def _embed_u8(x,bits,a,b,h,ch,packed,phases,write):
    H,W,C=x.shape;nc=W//2;K=H*nc;N=(bits.size+1)//2
    a=a%K;j=b%K;clock=0;sse=np.int64(0)
    for i in range(N):
        row=j//nc;col=2*(j%nc)
        x0=int(x[row,col,ch]);x1=int(x[row,col+1,ch])
        code=(int(bits[2*i])<<1)|(int(bits[2*i+1]) if 2*i+1<bits.size else 0)
        u=int(phases[clock,code,0]);v=int(phases[clock,code,1])
        pair0=int(packed[u,x0]);pair1=int(packed[v,x1])
        v00=pair0&255;v01=pair1&255
        v10=pair0>>8;v11=pair1>>8
        d00=v00-x0;d01=v01-x1;d10=v10-x0;d11=v11-x1
        c0=d00*d00+d01*d01;c1=d10*d10+d11*d11
        if c0<=c1:cost=c0;y0=v00;y1=v01
        else:cost=c1;y0=v10;y1=v11
        sse+=cost
        if write:
            x[row,col,ch]=y0;x[row,col+1,ch]=y1
        j+=a
        if j>=K:j-=K
        clock+=1
        if clock>=h:clock=0
    return sse

@njit(cache=True)
def _nearest_general(v,r,P,M):
    lo=(-r+P-1)//P;hi=(M-r)//P
    if lo>hi:return -1
    k=(2*(v-r)+P)//(2*P)
    if k<lo:k=lo
    if k>hi:k=hi
    return r+P*k

@njit(cache=True)
def _embed_uint16(x,bits,seed,a,b,stride,h,ch,write):
    H,W,C=x.shape;nc=W//2;K=H*nc;N=(bits.size+1)//2
    P=2*h;M=65535;sse=np.int64(0);j=b%K;aa=a%K;clock=0
    for i in range(N):
        row=j//nc;col=2*(j%nc)
        v0=int(x[row,col,ch]);v1=int(x[row,col+1,ch])
        z0=int(bits[2*i]);z1=int(bits[2*i+1]) if 2*i+1<bits.size else 0
        ps=2*((seed+stride*clock)%h);pd=2*((7*seed+(stride+2)*clock)%h)
        u=ps+z0*h;v=pd+z1*h
        r0=((u+v)//2)%P;r1=((u-v)//2)%P
        z00=_nearest_general(v0,r0,P,M);z01=_nearest_general(v1,r1,P,M)
        r0b=(r0+h)%P;r1b=(r1+h)%P
        z10=_nearest_general(v0,r0b,P,M);z11=_nearest_general(v1,r1b,P,M)
        c0=(z00-v0)**2+(z01-v1)**2 if z00>=0 and z01>=0 else np.int64(1<<60)
        c1=(z10-v0)**2+(z11-v1)**2 if z10>=0 and z11>=0 else np.int64(1<<60)
        if c0<=c1:cost=c0;y0=z00;y1=z01
        else:cost=c1;y0=z10;y1=z11
        if cost>=1<<60:return np.int64(-1)
        sse+=cost
        if write:
            x[row,col,ch]=y0;x[row,col+1,ch]=y1
        j+=aa
        if j>=K:j-=K
        clock+=1
        if clock>=h:clock=0
    return sse

@njit(cache=True)
def _extract(x,length,seed,a,b,stride,h,ch):
    H,W,C=x.shape;nc=W//2;K=H*nc;N=(length+1)//2;P=2*h
    out=np.empty(length,np.uint8);j=b%K;aa=a%K;clock=0
    for i in range(N):
        row=j//nc;col=2*(j%nc)
        u=int(x[row,col,ch]);v=int(x[row,col+1,ch])
        ps=2*((seed+stride*clock)%h);pd=2*((7*seed+(stride+2)*clock)%h)
        out[2*i]=(((u+v-ps+h//2)%P)//h)
        if 2*i+1<length:out[2*i+1]=(((u-v-pd+h//2)%P)//h)
        j+=aa
        if j>=K:j-=K
        clock+=1
        if clock>=h:clock=0
    return out

def embed(image,bits,*,h=None,seed=2026,psnr_floor=55.,channel=None,inplace=False):
    """Embed exact 2D lattice symbol; verify the actual image-wide PSNR floor.

    `h=None` chooses an analytic initial even spacing, then backs off if needed.
    This is not a globally optimal spacing search. Explicit h is supported for
    benchmark fairness and application-specific choices.
    """
    x,gray=_view(image)
    z=np.asarray(bits)
    if z.size==0 or not np.all((z==0)|(z==1)):raise ValueError('Raw binary payload required')
    bits=np.ascontiguousarray(z.reshape(-1),dtype=np.uint8)
    ch=(0 if x.shape[2]==1 else 1) if channel is None else int(channel)
    if not (0<=ch<x.shape[2]):raise ValueError('Invalid channel')
    a,b,stride=schedule(x.shape,len(bits),seed)
    M=int(np.iinfo(x.dtype).max)
    floor=float(psnr_floor)
    if not math.isfinite(floor) or floor<=0 or floor>120:raise ValueError('invalid PSNR floor')
    budget=math.floor(x.size*M*M*10**(-floor/10))
    def oracle(step,write,arr):
        if x.dtype==np.uint8:
            return int(_embed_u8(arr,bits,a,b,step,ch,_packed_lut(step),_residue_table(step,seed,stride),write))
        return int(_embed_uint16(arr,bits,seed,a,b,stride,step,ch,write))
    # Single-pass speculative embedding for copy-output: the exact projector
    # accumulates true SSE while embedding, so there is no need to run a
    # separate cost-oracle pass for a candidate that succeeds. A failed trial
    # is discarded (never returned) and a smaller even step is tried.
    if h is None and not inplace:
        N=(len(bits)+1)//2
        h0=2*max(1,math.floor(math.sqrt(3.*budget/N)/2))
        cap=64 if x.dtype==np.uint8 else 32768
        trial=min(cap,h0)
        while trial>=2:
            candidate=x.copy()
            cost=oracle(trial,True,candidate)
            if 0<=cost<=budget:
                meta=Metadata(tuple(np.asarray(image).shape),len(bits),x.dtype.name,ch,trial,int(seed),floor)
                return (candidate[:,:,0] if gray else candidate),meta
            if M==255:
                trial-=2
            else:
                next_trial=max(2,2*(math.floor(trial*0.85)//2))
                trial=(next_trial if next_trial<trial else trial-2)
        raise ValueError('No tested even spacing satisfies PSNR floor')
    if h is None:
        N=(len(bits)+1)//2
        h0=2*max(1,math.floor(math.sqrt(3.*budget/N)/2))
        cap=64 if x.dtype==np.uint8 else 32768
        trial=min(cap,h0)
        while trial>=2:
            c=oracle(trial,False,x)
            if 0<=c<=budget:
                h=trial;break
            # Small integer alphabet: examine each smaller even spacing.
            # For uint16 use geometric backoff to avoid thousands of passes.
            if M==255:
                trial-=2
            else:
                trial=max(0,2*(math.floor(trial*0.85)//2))
                if trial<2:trial=2
                if trial==2 and oracle(2,False,x)>budget:break
        if h is None:raise ValueError('No tested even spacing satisfies PSNR floor')
    h=int(h)
    if not (h>=2 and h%2==0 and (h<=64 if M==255 else h<=32768)):
        raise ValueError('h must be even, between 2 and the supported dynamic-range limit')
    # Strict transactional safety: for inplace requested output, check before mutation.
    if inplace:
        cost=oracle(h,False,x)
        if cost<0 or cost>budget:raise ValueError('PSNR floor not feasible')
        y=x
    else:
        y=x.copy()
    sse=oracle(h,True,y)
    if sse<0 or sse>budget:raise ValueError('PSNR floor not feasible')
    if inplace and sse!=cost:raise ArithmeticError('cost-only oracle disagreed')
    meta=Metadata(tuple(np.asarray(image).shape),len(bits),x.dtype.name,ch,h,int(seed),floor)
    return (y[:,:,0] if gray else y),meta

def extract(image,metadata):
    info=Metadata.from_json(metadata) if isinstance(metadata,str) else metadata
    if not isinstance(info,Metadata):raise ValueError('metadata object required')
    x,_=_view(image)
    if tuple(np.asarray(image).shape)!=info.shape or x.dtype.name!=info.dtype:
        raise ValueError('image shape/dtype mismatch')
    if info.h<2 or info.h%2:raise ValueError('invalid h')
    a,b,stride=schedule(x.shape,info.length,info.seed)
    return _extract(x,info.length,info.seed,a,b,stride,info.h,info.channel)

def psnr(x,y):
    xa=np.asarray(x);ya=np.asarray(y)
    if xa.shape!=ya.shape or xa.dtype!=ya.dtype:raise ValueError('shape or dtype mismatch')
    M=np.iinfo(xa.dtype).max
    v=xa.astype(np.float64)-ya.astype(np.float64)
    sse=float(np.sum(v*v))
    return float('inf') if sse==0 else 10*math.log10(xa.size*float(M*M)/sse)
