from __future__ import annotations
import argparse, copy, json
from pathlib import Path

try:
    from .train import load_cfg, train
except ImportError:
    from train import load_cfg, train


def parse_floats(text: str):
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def main():
    ap = argparse.ArgumentParser(description='Reward coefficient sweep for PyTorch MAPPO.')
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--episodes', type=int, default=120)
    ap.add_argument('--p-inter-soft', default='-1,-3,-5')
    ap.add_argument('--p-move', default='-0.02,-0.05')
    ap.add_argument('--outroot', default=str(Path(__file__).with_name('outputs_sweep')))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    base_cfg = load_cfg(args.config)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for p_inter in parse_floats(args.p_inter_soft):
        for p_move in parse_floats(args.p_move):
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault('reward', {})['p_inter_soft'] = p_inter
            cfg.setdefault('reward', {})['p_move'] = p_move
            tag = f"pInter{str(p_inter).replace('-', 'm').replace('.', 'p')}_pMove{str(p_move).replace('-', 'm').replace('.', 'p')}"
            outdir = outroot / tag
            result = train(cfg, episodes=args.episodes, outdir=str(outdir), device=args.device)
            seen = result.get('seen', {})
            unseen = result.get('unseen', {})
            m_seen, n_seen = seen.get('mappo', {}), seen.get('nearest', {})
            m_unseen, n_unseen = unseen.get('mappo', {}), unseen.get('nearest', {})
            row = {
                'pInterSoft': p_inter,
                'pMove': p_move,
                'episodes': args.episodes,
                'outdir': str(outdir),
                'seenMakespan': m_seen.get('makespan'),
                'seenNearestMakespan': n_seen.get('makespan'),
                'seenImproveVsNearestPct': ((n_seen.get('makespan', 0) - m_seen.get('makespan', 0)) / n_seen.get('makespan', 1) * 100) if n_seen.get('makespan') else None,
                'seenSoft': m_seen.get('soft'),
                'seenHardExecuted': m_seen.get('hardExecuted'),
                'seenHardMask': m_seen.get('hardMask'),
                'seenMove': m_seen.get('move'),
                'unseenMakespan': m_unseen.get('makespan'),
                'unseenNearestMakespan': n_unseen.get('makespan'),
                'unseenImproveVsNearestPct': ((n_unseen.get('makespan', 0) - m_unseen.get('makespan', 0)) / n_unseen.get('makespan', 1) * 100) if n_unseen.get('makespan') else None,
                'unseenSoft': m_unseen.get('soft'),
                'unseenHardExecuted': m_unseen.get('hardExecuted'),
                'unseenHardMask': m_unseen.get('hardMask'),
                'unseenMove': m_unseen.get('move'),
            }
            rows.append(row)
            (outroot / 'reward_sweep_summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
