from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

try:
    from .env import CraneSchedulingEnv
    from .mappo import MAPPOAgent, Transition
    from .train import load_cfg, summarize, evaluate, _rows_to_samples
except ImportError:
    from env import CraneSchedulingEnv
    from mappo import MAPPOAgent, Transition
    from train import load_cfg, summarize, evaluate, _rows_to_samples


DEFAULT_LEVELS = [
    {"name": "L1_2C12L", "num_cranes": 2, "num_lifts": 12, "max_steps": 140},
    {"name": "L2_3C24L", "num_cranes": 3, "num_lifts": 24, "max_steps": 240},
    {"name": "L3_4C36L", "num_cranes": 4, "num_lifts": 36, "max_steps": 380},
]


def clone_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


def apply_level(base_cfg: Dict[str, Any], level: Dict[str, Any], base_seed: int, seed_runs: int, episodes: int) -> Dict[str, Any]:
    cfg = clone_cfg(base_cfg)
    cfg.update({
        "num_cranes": int(level["num_cranes"]),
        "num_lifts": int(level["num_lifts"]),
        "max_steps": int(level.get("max_steps", cfg.get("max_steps", 220))),
        "base_seed": base_seed,
        "seed_runs": seed_runs,
        "train_episodes": episodes,
    })
    return cfg


def checkpoint_state(agent: MAPPOAgent) -> Dict[str, Any]:
    return {
        "actor": copy.deepcopy(agent.actor.state_dict()),
        "critic": copy.deepcopy(agent.critic.state_dict()),
        "stats": copy.deepcopy(agent.stats),
    }


def restore_checkpoint(agent: MAPPOAgent, state: Dict[str, Any]):
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    agent.stats = state.get("stats", agent.stats)


def train_one_level(
    env: CraneSchedulingEnv,
    agent: MAPPOAgent,
    cfg: Dict[str, Any],
    episodes: int,
    level_name: str,
    seen_start: int,
    seen_count: int,
    eval_interval: int,
    select_best_checkpoint: bool = True,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any] | None]:
    m = cfg.get("mappo", {})
    eps_start = float(m.get("eps_start", 0.35))
    eps_end = float(m.get("eps_end", 0.05))
    seed_runs = int(cfg.get("seed_runs", 10))
    base = int(cfg.get("base_seed", 101))
    train_log: List[Dict[str, Any]] = []
    best_eval: Dict[str, Any] | None = None
    best_state: Dict[str, Any] | None = None
    for ep in range(1, episodes + 1):
        seed = base + ((ep - 1) % max(1, seed_runs))
        obs, masks, glob = env.reset(seed)
        transitions: List[Transition] = []
        ep_reward = 0.0
        done = False
        while not done:
            eps = eps_start + (eps_end - eps_start) * (ep / max(1, episodes))
            actions, logps = agent.act(obs, masks, greedy=False, epsilon=eps)
            values = agent.value(np.stack([glob] * env.nC))
            prev_obs, prev_masks, prev_glob = obs.copy(), masks.copy(), glob.copy()
            obs, masks, glob, rewards, done, info = env.step(actions)
            ep_reward += float(np.sum(rewards))
            for ci in range(env.nC):
                transitions.append(Transition(ci, prev_obs[ci], prev_masks[ci], prev_glob, int(actions[ci]), float(logps[ci]), float(values[ci]), float(rewards[ci]), bool(done)))
        agent.update(transitions)
        s = env.summary()
        row = {
            "level": level_name,
            "ep": ep,
            "seed": seed,
            "makespan": s["makespan"],
            "done": s["done"],
            "total": s["total"],
            "reward": ep_reward,
            "soft": s["softInter"],
            "hardExecuted": s.get("hardExecuted", 0),
            "hardMask": s.get("hardMask", 0),
            "travel": s["travelTotal"],
            "setup": s["setupTotal"],
            "move": s["moveTotal"],
        }
        train_log.append(row)
        if select_best_checkpoint and (ep % max(1, eval_interval) == 0 or ep == episodes):
            probe = evaluate(CraneSchedulingEnv(cfg), agent, seen_start, seen_count)["mappo"]
            probe = dict(probe, ep=ep, level=level_name)
            if best_eval is None or probe["makespan"] < best_eval["makespan"]:
                best_eval = probe
                best_state = checkpoint_state(agent)
        if ep % max(1, episodes // 5) == 0:
            print(f"[{level_name}] ep={ep}/{episodes} makespan={s['makespan']:.1f} done={s['done']}/{s['total']} hardExecuted={s.get('hardExecuted',0)} seenBest={best_eval['makespan'] if best_eval else None} updates={agent.stats['updates']}", flush=True)
    return train_log, (best_eval or {}), best_state


def evaluate_level(agent: MAPPOAgent, cfg: Dict[str, Any], seen_start: int, seen_count: int, unseen_start: int, unseen_count: int) -> Dict[str, Any]:
    env = CraneSchedulingEnv(cfg)
    return {
        "seen": evaluate(env, agent, seen_start, seen_count),
        "unseen": evaluate(env, agent, unseen_start, unseen_count),
        "representative": {
            "mappo": env.run_policy("mappo", model=agent, seed=seen_start, greedy=True),
            "nearest": env.run_policy("nearest", model=agent, seed=seen_start, greedy=True),
        },
    }


def write_curve_csv(path: Path, rows: List[Dict[str, Any]]):
    cols = ["level", "ep", "seed", "makespan", "done", "total", "reward", "soft", "hardExecuted", "hardMask", "travel", "setup", "move"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Curriculum PyTorch MAPPO: 2C/12L -> 3C/24L -> 4C/36L while preserving weights.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--episodes", nargs="+", type=int, default=[60, 90, 120], help="episodes per level")
    ap.add_argument("--outdir", default=str(Path(__file__).with_name("outputs_curriculum")))
    ap.add_argument("--base-seed", type=int, default=101)
    ap.add_argument("--seed-runs", type=int, default=10)
    ap.add_argument("--seen-start", type=int, default=101)
    ap.add_argument("--seen-count", type=int, default=10)
    ap.add_argument("--unseen-start", type=int, default=301)
    ap.add_argument("--unseen-count", type=int, default=20)
    ap.add_argument("--eval-interval", type=int, default=0, help="episodes between level checkpoint probes; 0 means episodes//5")
    ap.add_argument("--no-best-checkpoint", action="store_true", help="do not restore the best seen checkpoint at the end of each level")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base_cfg = load_cfg(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    levels = copy.deepcopy(DEFAULT_LEVELS)
    if len(args.episodes) != len(levels):
        raise SystemExit(f"--episodes must have {len(levels)} values")

    torch.manual_seed(args.base_seed)
    np.random.seed(args.base_seed)

    # Initialize on level 1; dimensions are intentionally level-invariant.
    cfg0 = apply_level(base_cfg, levels[0], args.base_seed, args.seed_runs, args.episodes[0])
    env0 = CraneSchedulingEnv(cfg0)
    obs, masks, glob = env0.reset(args.base_seed)
    agent = MAPPOAgent(obs_dim=obs.shape[-1], state_dim=glob.shape[-1], cfg=cfg0, device=args.device)

    t0 = time.time()
    all_curve: List[Dict[str, Any]] = []
    level_results: List[Dict[str, Any]] = []
    for level, episodes in zip(levels, args.episodes):
        cfg = apply_level(base_cfg, level, args.base_seed, args.seed_runs, episodes)
        # Keep latest cfg on the agent so PPO hyperparameters and saved checkpoint reflect the current stage.
        agent.cfg = cfg
        env = CraneSchedulingEnv(cfg)
        eval_interval = args.eval_interval or max(1, episodes // 5)
        curve, best_eval, best_state = train_one_level(
            env, agent, cfg, episodes, level["name"],
            args.seen_start, args.seen_count, eval_interval,
            select_best_checkpoint=not args.no_best_checkpoint,
        )
        if best_state is not None and not args.no_best_checkpoint:
            restore_checkpoint(agent, best_state)
        all_curve.extend(curve)
        ev = evaluate_level(agent, cfg, args.seen_start, args.seen_count, args.unseen_start, args.unseen_count)
        level_model_path = outdir / f"{level['name']}_best_model.pt"
        agent.save(str(level_model_path), extra={"level": level, "episodes": episodes, "bestCheckpointSeen": best_eval})
        item = {
            "level": level,
            "episodes": episodes,
            "config": cfg,
            "levelModelPath": str(level_model_path),
            "trainSummary": {
                "first10": summarize(_rows_to_samples(curve[:10])),
                "last20": summarize(_rows_to_samples(curve[-20:])),
                "best": min(curve, key=lambda r: r["makespan"]),
                "bestCheckpointSeen": best_eval,
            },
            **ev,
        }
        level_results.append(item)
        (outdir / "curriculum_mappo_result.json").write_text(json.dumps({"levels": level_results, "learningCurve": all_curve, "elapsedSec": time.time() - t0}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_curve_csv(outdir / "curriculum_learning_curve.csv", all_curve)

    model_path = outdir / "curriculum_mappo_model.pt"
    agent.save(str(model_path), extra={"curriculumLevels": levels, "episodes": args.episodes})
    final = {"levels": level_results, "learningCurve": all_curve, "elapsedSec": time.time() - t0, "modelPath": str(model_path), "modelStats": agent.stats}
    (outdir / "curriculum_mappo_result.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
