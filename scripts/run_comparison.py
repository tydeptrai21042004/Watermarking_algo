#!/usr/bin/env python3
"""DIAL-QIM + exactly five paper-equation baseline adapters, 60 attacks.

Baseline hyperparameters are tuned using a separate four-image DEVELOPMENT set,
never the eight evaluation hosts or their attacks. The DCT baseline adapters
are explicitly NOT claimed to reproduce the original papers' image experiments.
Full runs generate their own CSVs; prior results are NOT shipped in this ZIP.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from PIL import Image
from src.dial_qim import embed as proposal_embed,extract as proposal_extract,psnr
from src.attack_suite60 import attack_rgb,canonical_conditions,protocol_sha256,VERSION
from baselines.registry import METHODS as BASELINES, PAPERS,embed as baseline_embed, extract as baseline_extract

PROPOSAL='DIAL-QIM'
METHODS=(PROPOSAL,*BASELINES)
DEV=('house','pepper','sailboat','safari')
TEST=('airplane','athens','baboon','butfly','Girl','manhatan','milkdrop','tiffany')
# Predeclared grids: no attack-conditioned tuning, no test-image tuning.
GRIDS={
    'Scalar-QIM':(4,6,8,10,12,14,16,20,24,28,32),
    'STDM-QIM':(2,3,4,6,8,10,12,16,20,24,32,40),
    'SCS':(4,6,8,10,12,14,16,20,24,28,32),
    'ISS':(0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,8.,10.),
    'SS':(0.5,0.75,1.,1.5,2.,3.,4.,5.,6.,8.,10.),
}

def seed_for(*args):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:4],'little')

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)

def load_host(name):
    with Image.open(ROOT/'data/host'/f'{name}.bmp') as f:
        x=np.asarray(f.convert('RGB'),dtype=np.uint8).copy()
    if x.shape!=(512,512,3):raise ValueError('Common baseline adapter requires 512x512 RGB')
    return x

def bits_for(host,seed,length=4096):
    if length!=4096:raise ValueError('Paper-adapter comparison uses 4096 bits; proposal generality tested separately')
    return np.random.default_rng(seed_for('payload',host,seed)).integers(0,2,length,dtype=np.uint8)

def tune_baselines(names,seed,psnr_floor,dev_hosts,out):
    # Tune on distinct DEVELOPMENT hosts only, using clean PSNR and clean BER.
    selected={};rows=[]
    for name in names:
        if name==PROPOSAL:continue
        opts=[]
        for strength in GRIDS[name]:
            probe=[]
            for host in dev_hosts:
                img=load_host(host);bits=bits_for(host,seed)
                try:
                    y,key=baseline_embed(name,img,bits,step=strength,seed=seed_for(seed,host,name))
                    ber=float(np.mean(baseline_extract(name,y,key)!=bits))
                    quality=psnr(img,y)
                    row=dict(method=name,strength=strength,dev_host=host,psnr_db=quality,clean_ber=ber,status='ok')
                    probe.append(row)
                except Exception as exc:
                    row=dict(method=name,strength=strength,dev_host=host,status='error',error=f'{type(exc).__name__}: {exc}')
                rows.append(row)
            if len(probe)==len(dev_hosts):
                minpsnr=min(x['psnr_db'] for x in probe)
                meanpsnr=float(np.mean([x['psnr_db'] for x in probe]))
                meanber=float(np.mean([x['clean_ber'] for x in probe]))
                opts.append(dict(strength=strength,min_psnr=minpsnr,mean_psnr=meanpsnr,
                                 mean_clean_ber=meanber,dev_psnr_feasible=minpsnr>=psnr_floor))
        # Require 55 dB floor on the entire dev set; do not silently select invalid values.
        feasible=[x for x in opts if x['dev_psnr_feasible']]
        if not feasible:
            selected[name]={'status':'no_dev_feasible_strength','tested_strengths':list(GRIDS[name])}
        else:
            # Prefer clean recovery first, then highest strength while meeting
            # the development distortion floor. Any nonzero BER remains reported.
            best=min(feasible,key=lambda x:(x['mean_clean_ber'],abs(x['mean_psnr']-psnr_floor)))
            selected[name]={'status':'selected',**best}
        print('TUNING',name,selected[name],flush=True)
    write_csv(out/'development_tuning.csv',rows)
    (out/'chosen_strengths.json').write_text(json.dumps(selected,indent=2))
    return selected

def run(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    ap.add_argument('--hosts',nargs='+',choices=TEST,default=list(TEST))
    ap.add_argument('--dev-hosts',nargs='+',choices=DEV,default=list(DEV))
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--psnr-floor',type=float,default=55.)
    ap.add_argument('--attacks',choices=('all','focused','none'),default='all')
    ap.add_argument('--quick',action='store_true',help='One dev host, one test host, three attacks; smoke test only')
    ap.add_argument('--output',default='results/comparison')
    args=ap.parse_args(argv)
    names=tuple(dict.fromkeys(args.methods))
    hosts=args.hosts[:1] if args.quick else args.hosts
    dev=args.dev_hosts[:1] if args.quick else args.dev_hosts
    conds=canonical_conditions()[1:]
    if args.quick:conds=conds[:3]
    elif args.attacks=='focused':conds=[c for c in conds if c['attack_index'] in (1,2,13,14,25,26,37,38,49,50)]
    elif args.attacks=='none':conds=[]
    out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    selected=tune_baselines(names,args.seed,args.psnr_floor,dev,out)
    manifest={'methods':list(names),'baseline_papers':{k:PAPERS[k] for k in BASELINES},
              'development_hosts':list(dev),'test_hosts':list(hosts),'test_conditions':[c['condition_id'] for c in conds],
              'payload_bits':4096,'seed':args.seed,'psnr_floor_db':args.psnr_floor,
              'attack_protocol':VERSION,'attack_sha256':protocol_sha256(),
              'baseline_reproduction':'paper-equation adaptations, not original authors complete code',
              'runtime_contract':'single algorithm call includes output copy; JIT warmup excluded',
              'python':sys.version,'numpy':np.__version__,'platform':platform.platform()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    host_rows=[];attacked_rows=[]
    for host in hosts:
        x=load_host(host);bits=bits_for(host,args.seed)
        for name in names:
            rec=dict(host=host,method=name)
            try:
                seed=seed_for(args.seed,host,name)
                if name==PROPOSAL:
                    do_embed=lambda:proposal_embed(x,bits,seed=seed,psnr_floor=args.psnr_floor)
                    do_decode=lambda image,key:proposal_extract(image,key)
                    strength='auto'
                else:
                    s=selected[name]
                    if s['status']!='selected':raise ValueError('No PSNR-feasible development-set strength')
                    strength=s['strength']
                    do_embed=lambda:baseline_embed(name,x,bits,seed=seed,step=strength)
                    do_decode=lambda image,key:baseline_extract(name,image,key)
                # Warm-up excludes JIT compilation and unrelated cold costs.
                do_embed()
                start=time.perf_counter_ns();y,key=do_embed();embed_ms=(time.perf_counter_ns()-start)/1e6
                start=time.perf_counter_ns();clean=do_decode(y,key)
                decode_ms=(time.perf_counter_ns()-start)/1e6
                quality=float(psnr(x,y));clean_ber=float(np.mean(clean!=bits))
                # If the test host fails a development-calibrated requirement,
                # explicitly report it. No per-test retuning to hide failure.
                qualified=(quality>=args.psnr_floor-1e-9 and clean_ber==0)
                rec.update(status='qualified' if qualified else 'constraint_violation',strength=strength,
                           psnr_db=quality,clean_ber=clean_ber,embed_ms=embed_ms,
                           extract_ms=decode_ms)
                for cond in conds:
                    attacked=attack_rgb(y,cond,seed_for('attack',host,args.seed,cond['condition_id']))
                    res=do_decode(attacked,key)
                    attacked_rows.append({**rec,'condition_id':cond['condition_id'],
                                          'family':cond['category'],'attack':cond['attack'],
                                          'level':cond['level'],'ber':float(np.mean(res!=bits))})
            except Exception as exc:
                rec.update(status='error',error=f'{type(exc).__name__}: {exc}')
            host_rows.append(rec)
            print(host,name,rec['status'],rec.get('psnr_db',''),flush=True)
        write_csv(out/'host_metrics.csv',host_rows)
        write_csv(out/'per_attack.csv',attacked_rows)
    write_csv(out/'host_metrics.csv',host_rows)
    write_csv(out/'per_attack.csv',attacked_rows)
    print('Finished',len(host_rows),'host-method combinations; attack rows',len(attacked_rows))

if __name__=='__main__':run()
