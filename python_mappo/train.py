from __future__ import annotations
import argparse, json, time, copy
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
import numpy as np
import torch

try:
    from .env import CraneSchedulingEnv
    from .mappo import MAPPOAgent, Transition
except ImportError:
    from env import CraneSchedulingEnv
    from mappo import MAPPOAgent, Transition


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def summarize(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    def avg(key): return float(np.mean([s.get(key,0) for s in samples])) if samples else 0.0
    def sd(key): return float(np.std([s.get(key,0) for s in samples])) if samples else 0.0
    return {
        'n': len(samples),
        'completeRate': float(np.mean([100*s['done']/max(1,s['total']) for s in samples])) if samples else 0.0,
        'makespan': avg('makespan'), 'makespanSd': sd('makespan'), 'reward': avg('reward'),
        'soft': avg('softInter'),
        'hardExecuted': avg('hardExecuted'),
        'hardMask': avg('hardMask') if any('hardMask' in s for s in samples) else avg('hardInter'),
        'travel': avg('travelTotal'),
        'setup': avg('setupTotal'), 'move': avg('moveTotal'),
        'actualLiftRadius': avg('actualLiftRadiusAvg'), 'actualDangerRadius': avg('actualDangerRadiusAvg'),
    }


def evaluate(env: CraneSchedulingEnv, agent: MAPPOAgent, seed_start: int, seed_count: int) -> Dict[str, Dict[str, float]]:
    policies = ['mappo', 'nearest', 'radiusPriority', 'random']
    out={}
    for pol in policies:
        samples=[]
        for seed in range(seed_start, seed_start+seed_count):
            samples.append(env.run_policy(pol, model=agent, seed=seed, greedy=True))
        out[pol]=summarize(samples)
    return out


def train(cfg: Dict[str, Any], episodes: Optional[int] = None, outdir: str = 'outputs', device: str = 'cpu') -> Dict[str, Any]:
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    env=CraneSchedulingEnv(cfg)
    torch.manual_seed(int(cfg.get('base_seed',101)))
    np.random.seed(int(cfg.get('base_seed',101)))
    obs,masks,glob=env.reset(cfg.get('base_seed',101))
    agent=MAPPOAgent(obs_dim=obs.shape[-1], state_dim=glob.shape[-1], cfg=cfg, device=device)
    m=cfg.get('mappo', {})
    episodes = int(episodes or cfg.get('train_episodes',150))
    seed_runs=int(cfg.get('seed_runs',10)); base=int(cfg.get('base_seed',101))
    eps_start=float(m.get('eps_start',0.35)); eps_end=float(m.get('eps_end',0.05))
    train_log=[]; best=None; best_eval=None; best_state=None; t0=time.time()
    for ep in range(1, episodes+1):
        seed = base + ((ep-1) % max(1,seed_runs))
        obs,masks,glob=env.reset(seed)
        transitions=[]; ep_reward=0.0; done=False
        while not done:
            eps = eps_start + (eps_end-eps_start)*(ep/max(1,episodes))
            actions, logps = agent.act(obs, masks, greedy=False, epsilon=eps)
            values = agent.value(np.stack([glob]*env.nC))
            prev_obs, prev_masks, prev_glob = obs.copy(), masks.copy(), glob.copy()
            obs,masks,glob,rewards,done,info=env.step(actions)
            ep_reward += float(np.sum(rewards))
            for ci in range(env.nC):
                transitions.append(Transition(ci, prev_obs[ci], prev_masks[ci], prev_glob, int(actions[ci]), float(logps[ci]), float(values[ci]), float(rewards[ci]), bool(done)))
        loss=agent.update(transitions)
        summary=env.summary(); summary['reward']=ep_reward
        row={'ep':ep,'seed':seed,'makespan':summary['makespan'],'done':summary['done'],'total':summary['total'],'reward':ep_reward,'soft':summary['softInter'],'hardExecuted':summary.get('hardExecuted',0),'hardMask':summary.get('hardMask',summary.get('hardInter',0)),'move':summary['moveTotal'],'travel':summary['travelTotal'],'setup':summary['setupTotal']}
        train_log.append(row)
        if summary['done']==summary['total'] and (best is None or summary['makespan']<best['makespan']):
            best=dict(row)
        # Keep the best checkpoint by deterministic seen-seed makespan, not merely the final PPO weights.
        if ep % max(5, episodes//10) == 0 or ep == episodes:
            probe = evaluate(env, agent, int(cfg.get('seen_seed_start',101)), int(cfg.get('seen_seed_count',10)))['mappo']
            if best_eval is None or probe['makespan'] < best_eval['makespan']:
                best_eval = dict(probe, ep=ep)
                best_state = {
                    'actor': copy.deepcopy(agent.actor.state_dict()),
                    'critic': copy.deepcopy(agent.critic.state_dict()),
                    'stats': copy.deepcopy(agent.stats),
                }
        if ep % max(1, episodes//10) == 0:
            print(f"ep={ep}/{episodes} makespan={summary['makespan']:.1f} reward={ep_reward:.1f} best={best['makespan'] if best else None} seenBest={best_eval['makespan'] if best_eval else None} updates={agent.stats['updates']}", flush=True)
    if best_state is not None:
        agent.actor.load_state_dict(best_state['actor'])
        agent.critic.load_state_dict(best_state['critic'])
        agent.stats = best_state['stats']
    seen=evaluate(env, agent, int(cfg.get('seen_seed_start',101)), int(cfg.get('seen_seed_count',10)))
    unseen=evaluate(env, agent, int(cfg.get('unseen_seed_start',201)), int(cfg.get('unseen_seed_count',30)))
    model_path=outdir/'pytorch_mappo_model.pt'
    agent.save(str(model_path), extra={'best':best})
    representative={
        'mappo': env.run_policy('mappo', model=agent, seed=int(cfg.get('seen_seed_start',101)), greedy=True),
        'nearest': env.run_policy('nearest', model=agent, seed=int(cfg.get('seen_seed_start',101)), greedy=True),
    }
    result={'config':cfg,'elapsedSec':time.time()-t0,'trainEpisodes':episodes,'trainSummary':{'best':best,'bestCheckpointSeen':best_eval,'first10':summarize(_rows_to_samples(train_log[:10])),'last20':summarize(_rows_to_samples(train_log[-20:])),'modelStats':agent.stats},'learningCurve':train_log,'seen':seen,'unseen':unseen,'representative':representative,'modelPath':str(model_path)}
    result_path=outdir/'pytorch_mappo_validation_result.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    curve_path=outdir/'learning_curve.csv'
    curve_cols=['ep','seed','makespan','done','total','reward','soft','hardExecuted','hardMask','travel','setup','move']
    curve_path.write_text(','.join(curve_cols)+'\n'+'\n'.join(','.join(str(r.get(c,'')) for c in curve_cols) for r in train_log), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _rows_to_samples(rows):
    return [{'done':r['done'],'total':r['total'],'makespan':r['makespan'],'reward':r['reward'],'softInter':r['soft'],'hardExecuted':r.get('hardExecuted',0),'hardMask':r.get('hardMask',0),'hardInter':r.get('hardMask',0),'moveTotal':r['move'],'travelTotal':r.get('travel',0),'setupTotal':r.get('setup',0)} for r in rows]


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--episodes', type=int, default=None)
    ap.add_argument('--outdir', default=str(Path(__file__).with_name('outputs')))
    ap.add_argument('--device', default='cpu')
    args=ap.parse_args()
    train(load_cfg(args.config), args.episodes, args.outdir, args.device)
