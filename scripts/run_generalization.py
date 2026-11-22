#!/usr/bin/env python3
"""Verify the ONE proposal on different image sizes, channels and uint8/uint16 types."""
from __future__ import annotations
import csv,sys, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from PIL import Image
from src.dial_qim import embed,extract,psnr

CASES=((96,128,127),(180,300,777),(256,256,1024),(512,512,4096),(768,1024,10007))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',default='results/generalization.csv')
    args=p.parse_args()
    base=np.asarray(Image.open(ROOT/'data/host/airplane.bmp').convert('RGB'))
    rows=[]
    for h,w,length in (CASES[:2] if args.quick else CASES):
        rgb=cv2.resize(base,(w,h),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        formats={'RGB8':rgb,'GRAY8':gray,'RGBA8':np.concatenate([rgb,np.full((h,w,1),255,np.uint8)],2),
                 'GRAY16':gray.astype(np.uint16)*257}
        for fmt,x in formats.items():
            bits=np.random.default_rng(2026+length).integers(0,2,length,dtype=np.uint8)
            row={'format':fmt,'height':h,'width':w,'bits':length}
            try:
                y,meta=embed(x,bits,seed=2026,psnr_floor=55.)
                row.update(status='ok',h=meta.h,psnr_db=psnr(x,y),
                           clean_ber=float(np.mean(extract(y,meta)!=bits)))
                if row['clean_ber'] or row['psnr_db']<55:raise AssertionError('Invariant violated')
            except Exception as exc:
                row.update(status='infeasible_or_error',error=f'{type(exc).__name__}: {exc}')
            rows.append(row)
            print(row,flush=True)
    dest=ROOT/args.output;dest.parent.mkdir(parents=True,exist_ok=True)
    cols=list(dict.fromkeys(k for v in rows for k in v))
    with dest.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(rows)
if __name__=='__main__':main()
