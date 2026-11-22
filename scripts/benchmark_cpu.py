"""Warm CPU fair benchmark: exact DIAL vs faithful compiled Scalar-QIM adapter."""
from __future__ import annotations
import sys,time,statistics,json,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from numba import njit
from PIL import Image
from src import dial_qim as D
from baselines import scalar_qim as scalar

@njit(cache=True)
def scalar_compiled_copy(x,bits,positions,phases,step):
    y=x.copy()
    for i in range(bits.size):
        p=int(positions[i]);row=(i//64)*8+p//24
        col=(i%64)*8+(p//3)%8;ch=p%3
        v=int(y[row,col,ch]);residue=(int(phases[i])+int(bits[i])*(step//2))%step
        n=(2*(v-residue)+step)//(2*step)
        if n<0:n=0
        limit=(255-residue)//step
        if n>limit:n=limit
        y[row,col,ch]=residue+n*step
    return y

def med(fn,n=200):
    for _ in range(5):fn()
    res=[]
    for _ in range(n):
        start=time.perf_counter_ns();fn();res.append((time.perf_counter_ns()-start)/1e6)
    return float(statistics.median(res))

def main():
    rng=np.random.default_rng(2026);bits=rng.integers(0,2,4096,dtype=np.uint8)
    hosts=['airplane','athens','baboon','butfly','Girl','manhatan','milkdrop','tiffany']
    rows=[]
    for host in hosts:
        x=np.array(Image.open(ROOT/'data'/'host'/f'{host}.bmp').convert('RGB'))
        dial_y,metadata=D.embed(x,bits)
        admissible=[]
        for step in range(2,33,2):
            yy=scalar.embed(x,bits,step=step,seed=71)
            if scalar.psnr(x,yy)>=55 and np.array_equal(scalar.extract(yy,step=step,seed=71),bits):
                admissible.append((step,yy))
        if not admissible:raise RuntimeError('No scalar baseline PSNR-feasible step')
        step,baseline_y=max(admissible,key=lambda q:q[0]);pos,phase=scalar.schedule(71,step)
        assert np.array_equal(baseline_y,scalar_compiled_copy(x,bits,pos,phase,step))
        a,b,stride=D.schedule(x.shape,len(bits),2026)
        tbl=D._packed_lut(metadata.h);res=D._residue_table(metadata.h,2026,stride)
        inpl=x.copy()
        row={'host':host,'dial_h':metadata.h,'scalar_step':step,
          'dial_psnr':D.psnr(x,dial_y),'scalar_psnr':scalar.psnr(x,baseline_y),
          'dial_copy_ms':med(lambda:D.embed(x,bits,h=metadata.h)[0]),
          'scalar_copy_ms':med(lambda:scalar_compiled_copy(x,bits,pos,phase,step)),
          'dial_auto_ms':med(lambda:D.embed(x,bits)[0],n=75),
          'dial_inplace_kernel_ms':med(lambda:D._embed_u8(inpl,bits,a,b,metadata.h,1,tbl,res,True))}
        rows.append(row)
        print(host, 'DIAL copy',round(row['dial_copy_ms'],5),'Scalar',round(row['scalar_copy_ms'],5),
              'DIAL auto',round(row['dial_auto_ms'],5),flush=True)
    output=ROOT/'results'/'benchmark_cpu.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'rows':rows,'machine':platform.machine(),'cpu':platform.processor(),
        'note':'Warm JIT, 8-bit 512x512 RGB, 4096 raw bits, same independent-output contract'},indent=2))
    print('OUTPUT:',output)

if __name__=='__main__':main()
