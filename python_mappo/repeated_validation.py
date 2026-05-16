from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

try:
    from .train import load_cfg, train
except ImportError:
    from train import load_cfg, train


def pct_improve(base: float, value: float) -> float:
    if not base:
        return 0.0
    return (base - value) / base * 100.0


def safe_get(d: Dict[str, Any], *keys, default=0.0):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def row_from_result(label: str, cfg: Dict[str, Any], episodes: int, result: Dict[str, Any], outdir: Path) -> Dict[str, Any]:
    seen = result.get("seen", {})
    unseen = result.get("unseen", {})
    sm = seen.get("mappo", {})
    um = unseen.get("mappo", {})
    sn = seen.get("nearest", {})
    un = unseen.get("nearest", {})
    sr = seen.get("random", {})
    ur = unseen.get("random", {})
    return {
        "run_label": label,
        "episodes": episodes,
        "train_seed_start": cfg.get("base_seed"),
        "train_seed_count": cfg.get("seed_runs"),
        "seen_seed_start": cfg.get("seen_seed_start"),
        "seen_seed_count": cfg.get("seen_seed_count"),
        "unseen_seed_start": cfg.get("unseen_seed_start"),
        "unseen_seed_count": cfg.get("unseen_seed_count"),
        "cranes": cfg.get("num_cranes"),
        "lifts": cfg.get("num_lifts"),
        "candidate_k": cfg.get("candidate_k"),
        "seen_mappo_makespan_mean": sm.get("makespan", 0),
        "seen_mappo_makespan_std": sm.get("makespanSd", 0),
        "seen_nearest_makespan_mean": sn.get("makespan", 0),
        "seen_random_makespan_mean": sr.get("makespan", 0),
        "seen_nearest_improve_pct": pct_improve(sn.get("makespan", 0), sm.get("makespan", 0)),
        "seen_random_improve_pct": pct_improve(sr.get("makespan", 0), sm.get("makespan", 0)),
        "unseen_mappo_makespan_mean": um.get("makespan", 0),
        "unseen_mappo_makespan_std": um.get("makespanSd", 0),
        "unseen_nearest_makespan_mean": un.get("makespan", 0),
        "unseen_random_makespan_mean": ur.get("makespan", 0),
        "unseen_nearest_improve_pct": pct_improve(un.get("makespan", 0), um.get("makespan", 0)),
        "unseen_random_improve_pct": pct_improve(ur.get("makespan", 0), um.get("makespan", 0)),
        "completion_rate_seen": sm.get("completeRate", 0),
        "completion_rate_unseen": um.get("completeRate", 0),
        "hard_executed_seen": sm.get("hardExecuted", 0),
        "hard_executed_unseen": um.get("hardExecuted", 0),
        "hard_mask_seen": sm.get("hardMask", 0),
        "hard_mask_unseen": um.get("hardMask", 0),
        "soft_seen": sm.get("soft", 0),
        "soft_unseen": um.get("soft", 0),
        "travel_unseen": um.get("travel", 0),
        "setup_unseen": um.get("setup", 0),
        "move_unseen": um.get("move", 0),
        "generalization_gap": um.get("makespan", 0) - sm.get("makespan", 0),
        "model_path": result.get("modelPath", ""),
        "outdir": str(outdir),
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def vals(k):
        return [float(r.get(k, 0) or 0) for r in rows]
    def mean(k):
        v = vals(k)
        return sum(v) / len(v) if v else 0.0
    def sd(k):
        v = vals(k)
        if not v:
            return 0.0
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return {
        "runs": len(rows),
        "unseen_mappo_makespan_mean": mean("unseen_mappo_makespan_mean"),
        "unseen_mappo_makespan_std_across_runs": sd("unseen_mappo_makespan_mean"),
        "unseen_nearest_improve_pct_mean": mean("unseen_nearest_improve_pct"),
        "seen_mappo_makespan_mean": mean("seen_mappo_makespan_mean"),
        "seen_nearest_improve_pct_mean": mean("seen_nearest_improve_pct"),
        "completion_rate_unseen_mean": mean("completion_rate_unseen"),
        "hard_executed_unseen_mean": mean("hard_executed_unseen"),
        "hard_mask_unseen_mean": mean("hard_mask_unseen"),
        "generalization_gap_mean": mean("generalization_gap"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def write_report(path: Path, rows: List[Dict[str, Any]], agg: Dict[str, Any]):
    lines = [
        "# Stage-1 PyTorch MAPPO 반복 검증 요약",
        "",
        f"- 반복 run 수: {agg['runs']}",
        f"- Unseen MAPPO makespan 평균: {agg['unseen_mappo_makespan_mean']:.2f}",
        f"- Unseen MAPPO makespan run 간 표준편차: {agg['unseen_mappo_makespan_std_across_runs']:.2f}",
        f"- Unseen Nearest 대비 평균 개선율: {agg['unseen_nearest_improve_pct_mean']:.2f}%",
        f"- Seen Nearest 대비 평균 개선율: {agg['seen_nearest_improve_pct_mean']:.2f}%",
        f"- Unseen 완료율 평균: {agg['completion_rate_unseen_mean']:.2f}%",
        f"- Unseen 실행 hard overlap 평균: {agg['hard_executed_unseen_mean']:.2f}",
        f"- Unseen HardMask 평균: {agg['hard_mask_unseen_mean']:.2f}",
        "",
        "## Run별 결과",
        "",
    ]
    for r in rows:
        lines += [
            f"### {r['run_label']}",
            f"- train seeds: {r['train_seed_start']}~{int(r['train_seed_start']) + int(r['train_seed_count']) - 1}",
            f"- episodes: {r['episodes']}",
            f"- seen MAPPO makespan: {float(r['seen_mappo_makespan_mean']):.2f} / Nearest 개선율 {float(r['seen_nearest_improve_pct']):.2f}%",
            f"- unseen MAPPO makespan: {float(r['unseen_mappo_makespan_mean']):.2f} / Nearest 개선율 {float(r['unseen_nearest_improve_pct']):.2f}%",
            f"- hardExecuted unseen: {float(r['hard_executed_unseen']):.2f}, HardMask unseen: {float(r['hard_mask_unseen']):.2f}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Repeated seed-group validation for Stage-1 PyTorch MAPPO.")
    ap.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--outroot", default=str(Path(__file__).with_name("outputs_repeated")))
    ap.add_argument("--groups", nargs="+", default=["101:10:101:10:201:30", "111:10:111:10:231:30", "121:10:121:10:261:30"], help="base:trainCount:seenStart:seenCount:unseenStart:unseenCount")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    base_cfg = load_cfg(args.config)
    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, spec in enumerate(args.groups, start=1):
        base, train_count, seen_start, seen_count, unseen_start, unseen_count = [int(x) for x in spec.split(":")]
        cfg = json.loads(json.dumps(base_cfg))
        cfg.update({
            "base_seed": base,
            "seed_runs": train_count,
            "seen_seed_start": seen_start,
            "seen_seed_count": seen_count,
            "unseen_seed_start": unseen_start,
            "unseen_seed_count": unseen_count,
            "train_episodes": args.episodes,
        })
        label = f"group{idx}_train{base}-{base+train_count-1}_unseen{unseen_start}-{unseen_start+unseen_count-1}"
        outdir = outroot / label
        result = train(cfg, episodes=args.episodes, outdir=str(outdir), device=args.device)
        rows.append(row_from_result(label, cfg, args.episodes, result, outdir))
        write_csv(outroot / "repeated_experiment_summary.csv", rows)
        agg = aggregate(rows)
        (outroot / "repeated_experiment_summary.json").write_text(json.dumps({"runs": rows, "aggregate": agg}, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(outroot / "report_summary.md", rows, agg)
    print(json.dumps({"runs": rows, "aggregate": aggregate(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
