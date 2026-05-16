from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
try:
    from .env import CraneSchedulingEnv
    from .mappo import MAPPOAgent
    from .train import summarize
except ImportError:
    from env import CraneSchedulingEnv
    from mappo import MAPPOAgent
    from train import summarize


def load_cfg(path):
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--model', required=True)
    ap.add_argument('--seed-start', type=int, default=201)
    ap.add_argument('--seed-count', type=int, default=30)
    ap.add_argument('--out', default='')
    args=ap.parse_args()
    cfg=load_cfg(args.config); env=CraneSchedulingEnv(cfg); agent=MAPPOAgent.load(args.model)
    out={}
    representative={}
    for pol in ['mappo','nearest','radiusPriority','random']:
        samples=[]
        for s in range(args.seed_start,args.seed_start+args.seed_count):
            sample=env.run_policy(pol, model=agent, seed=s, greedy=True)
            samples.append(sample)
            if s == args.seed_start:
                representative[pol]=sample
        out[pol]=summarize(samples)
    text=json.dumps({'seedStart':args.seed_start,'seedCount':args.seed_count,'out':out,'representative':representative},ensure_ascii=False,indent=2)
    if args.out: Path(args.out).write_text(text,encoding='utf-8')
    print(text)
if __name__=='__main__': main()
