from __future__ import annotations
import argparse, json
from pathlib import Path

try:
    from .train import load_cfg, train
except ImportError:
    from train import load_cfg, train


def main():
    ap = argparse.ArgumentParser(description='Run repeated PyTorch MAPPO episode-count experiments.')
    ap.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    ap.add_argument('--episodes', nargs='+', type=int, default=[300, 500])
    ap.add_argument('--outroot', default=str(Path(__file__).with_name('outputs_experiments')))
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    summary = []
    for ep in args.episodes:
        outdir = outroot / f'ep{ep}'
        result = train(cfg, episodes=ep, outdir=str(outdir), device=args.device)
        row = {
            'episodes': ep,
            'outdir': str(outdir),
            'modelPath': result.get('modelPath'),
            'seen': result.get('seen', {}).get('mappo', {}),
            'unseen': result.get('unseen', {}).get('mappo', {}),
            'seenNearest': result.get('seen', {}).get('nearest', {}),
            'unseenNearest': result.get('unseen', {}).get('nearest', {}),
            'best': result.get('trainSummary', {}).get('best'),
            'bestCheckpointSeen': result.get('trainSummary', {}).get('bestCheckpointSeen'),
        }
        summary.append(row)
        (outroot / 'episode_experiment_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
