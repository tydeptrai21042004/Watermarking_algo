#!/usr/bin/env python3
"""Controlled ablations of ONE DIAL-QIM method; outputs CSV only.

Default: 8 held-out hosts x 5 ablation settings x 60 fixed attacks.
The same raw payload, exact quantizer step, seeds, and attacks are shared within
host. If a variant violates 55 dB, it is recorded as INFEASIBLE rather than
quietly given a weaker constraint or counted as a valid method.

A separate step-control experiment fixes h=8 instead of the automatically
selected h. This tests parameter choice without introducing another method.
"""
from __future__ import annotations
import argparse,csv,hashlib,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from PIL import Image
from src.dial_qim import embed,extract,psnr
from src.ablation_controls import VARIANTS,embed_variant,extract_variant
from src.attack_suite60 import canonical_conditions,attack_rgb,protocol_sha256

HOSTS=('airplane','athens','baboon','butfly','Girl','manhatan','milkdrop','tiffany')
EXPERIMENTS=VARIANTS+('fixed_h8',)

def seed_for(*terms):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,terms)).encode()).digest()[:4],'little')

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    names=list(dict.fromkeys(key for item in rows for key in item))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--hosts',nargs='+',default=list(HOSTS),choices=HOSTS)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--bits',type=int,default=4096)
    ap.add_argument('--psnr-floor',type=float,default=55.)
    ap.add_argument('--attacks',choices=('all','focused','none'),default='all')
    ap.add_argument('--output',default='results/ablations')
    ap.add_argument('--allow-infeasible',action='store_true',help='Diagnostic only; mark violations, never qualify them as valid')
    ap.add_argument('--retune-for-feasibility',action='store_true',
                    help='On clean host only, lower even h until 55 dB is met; report the changed h explicitly')
    args=ap.parse_args(argv)
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    conditions=canonical_conditions()[1:]
    if args.attacks=='focused':conditions=[c for c in conditions if c['attack_index'] in (1,2,13,14,25,26,37,38,49,50)]
    if args.attacks=='none':conditions=[]
    rows=[];host_rows=[]
    for host in args.hosts:
        x=np.array(Image.open(ROOT/'data/host'/f'{host}.bmp').convert('RGB'))
        bits=np.random.default_rng(seed_for('payload',host,args.seed)).integers(0,2,args.bits,dtype=np.uint8)
        yfull,full_meta=embed(x,bits,seed=args.seed,psnr_floor=args.psnr_floor)
        # Model strength fixed across component removals. h=8 is an additional
        # controlled parameter-selection ablation, not a distinct proposal.
        for variant in EXPERIMENTS:
            h=8 if variant=='fixed_h8' else full_meta.h
            record=dict(host=host,ablation=variant,seed=args.seed,payload_bits=args.bits,
                        h=h,protocol_hash=protocol_sha256())
            try:
                # Compile JIT kernels once; only warm execution is timed.
                embed_variant(x,bits,variant=('full' if variant=='fixed_h8' else variant),
                              h=h,seed=args.seed,psnr_floor=args.psnr_floor,strict=False)
                t0=time.perf_counter_ns()
                effective_variant=('full' if variant=='fixed_h8' else variant)
                try:
                    y,meta=embed_variant(x,bits,variant=effective_variant,
                                         h=h,seed=args.seed,psnr_floor=args.psnr_floor,
                                         strict=not args.allow_infeasible)
                except ValueError as exc:
                    if not args.retune_for_feasibility or variant in ('full','arithmetic_projection','fixed_h8'):
                        raise
                    feasible=False
                    for smaller_h in range(h-2,1,-2):
                        try:
                            y,meta=embed_variant(x,bits,variant=effective_variant,
                                                 h=smaller_h,seed=args.seed,psnr_floor=args.psnr_floor,
                                                 strict=True)
                            feasible=True
                            record.update(h=smaller_h,retuned_from=h)
                            break
                        except ValueError:
                            continue
                    if not feasible:raise exc
                elapsed=(time.perf_counter_ns()-t0)/1e6
                rec=extract_variant(y,meta)
                clean=float(np.mean(rec!=bits))
                quality=float(psnr(x,y))
                qualified=(quality>=args.psnr_floor-1e-10 and clean==0)
                record.update(status='qualified' if qualified else 'diagnostic_infeasible',
                              clean_ber=clean,psnr_db=quality,embed_ms=elapsed)
                if variant=='full':
                    if not np.array_equal(y,yfull) or not np.array_equal(rec,extract(yfull,full_meta)):
                        raise AssertionError('Ablation full path differs from production DIAL-QIM')
                if variant=='arithmetic_projection':
                    if not np.array_equal(y,yfull):
                        raise AssertionError('Direct arithmetic must reproduce the packed optimum exactly')
                for condition in conditions:
                    attacked=attack_rgb(y,condition,seed_for('attack',host,args.seed,condition['condition_id']))
                    decoded=extract_variant(attacked,meta)
                    rows.append({**record,'condition_id':condition['condition_id'],
                                 'family':condition['category'],'attack':condition['attack'],
                                 'level':condition['level'],'ber':float(np.mean(decoded!=bits))})
            except Exception as exc:
                record.update(status='infeasible_or_error',error=f'{type(exc).__name__}: {exc}')
            host_rows.append(record)
            print(host,variant,record['status'],f"PSNR={record.get('psnr_db','NA')}",flush=True)
        write_csv(out/'ablation_hosts.csv',host_rows)
        write_csv(out/'ablation_attacks.csv',rows)
    write_csv(out/'ablation_hosts.csv',host_rows)
    write_csv(out/'ablation_attacks.csv',rows)
    print('Wrote',out,'with',len(host_rows),'host configurations and',len(rows),'attack rows')

if __name__=='__main__': main()
