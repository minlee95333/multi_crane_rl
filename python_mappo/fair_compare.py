from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def pct_improve(base: float, value: float) -> float:
    return (base - value) / base * 100.0 if base else 0.0


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_browser(browser: Dict[str, Any], split: str, policy: str) -> Dict[str, Any]:
    src = browser.get(split, {}).get(policy, {})
    return {
        "engine": "browser-light",
        "split": split,
        "policy": policy,
        "makespan": src.get("makespan", 0),
        "makespanSd": src.get("makespanSd", 0),
        "completeRate": src.get("completeRate", src.get("completion", 0)),
        "soft": src.get("soft", 0),
        "hardExecuted": src.get("hardExecuted", 0),
        "hardMask": src.get("hardMask", src.get("hardInter", 0)),
        "travel": src.get("travel", 0),
        "setup": src.get("setup", 0),
        "move": src.get("move", 0),
    }


def norm_pytorch(result: Dict[str, Any], split: str, policy: str) -> Dict[str, Any]:
    src = result.get(split, {}).get(policy, {})
    name = {"mappo": "PyTorch MAPPO", "nearest": "Nearest", "radiusPriority": "Same-radius-priority", "random": "Random"}.get(policy, policy)
    return {
        "engine": "pytorch",
        "split": split,
        "policy": name,
        "makespan": src.get("makespan", 0),
        "makespanSd": src.get("makespanSd", 0),
        "completeRate": src.get("completeRate", 0),
        "soft": src.get("soft", 0),
        "hardExecuted": src.get("hardExecuted", 0),
        "hardMask": src.get("hardMask", src.get("hardInter", 0)),
        "travel": src.get("travel", 0),
        "setup": src.get("setup", 0),
        "move": src.get("move", 0),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Fair comparison report between browser-light MAPPO and PyTorch MAPPO summaries.")
    ap.add_argument("--browser", default="curriculum_runs/current_rl_validation_result.json")
    ap.add_argument("--pytorch", default="python_mappo/outputs_repeated/repeated_experiment_summary.json")
    ap.add_argument("--fallback-pytorch", default="python_mappo/outputs/pytorch_mappo_validation_result.json")
    ap.add_argument("--outdir", default="python_mappo/outputs_fair_compare")
    args = ap.parse_args()

    browser_path = Path(args.browser)
    pytorch_path = Path(args.pytorch)
    if not pytorch_path.exists():
        pytorch_path = Path(args.fallback_pytorch)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    browser = load_json(browser_path)
    pytorch_doc = load_json(pytorch_path)
    rows: List[Dict[str, Any]] = []

    browser_policies = ["RL-before", "MAPPO-trained", "Nearest", "Same-radius", "Random"]
    for split in ["seen", "unseen"]:
        for pol in browser_policies:
            if pol in browser.get(split, {}):
                rows.append(norm_browser(browser, split, pol))

    # If repeated summary is provided, include aggregate and each run's PyTorch MAPPO rows.
    if "runs" in pytorch_doc and "aggregate" in pytorch_doc:
        agg = pytorch_doc["aggregate"]
        rows.extend([
            {"engine": "pytorch-repeated-aggregate", "split": "seen", "policy": "PyTorch MAPPO", "makespan": agg.get("seen_mappo_makespan_mean", 0), "makespanSd": 0, "completeRate": 0, "soft": 0, "hardExecuted": 0, "hardMask": 0, "travel": 0, "setup": 0, "move": 0},
            {"engine": "pytorch-repeated-aggregate", "split": "unseen", "policy": "PyTorch MAPPO", "makespan": agg.get("unseen_mappo_makespan_mean", 0), "makespanSd": agg.get("unseen_mappo_makespan_std_across_runs", 0), "completeRate": agg.get("completion_rate_unseen_mean", 0), "soft": 0, "hardExecuted": agg.get("hard_executed_unseen_mean", 0), "hardMask": agg.get("hard_mask_unseen_mean", 0), "travel": 0, "setup": 0, "move": 0},
        ])
        for r in pytorch_doc["runs"]:
            rows.append({"engine": "pytorch-run", "split": "seen", "policy": r["run_label"], "makespan": r["seen_mappo_makespan_mean"], "makespanSd": r["seen_mappo_makespan_std"], "completeRate": r["completion_rate_seen"], "soft": r["soft_seen"], "hardExecuted": r["hard_executed_seen"], "hardMask": r["hard_mask_seen"], "travel": 0, "setup": 0, "move": 0})
            rows.append({"engine": "pytorch-run", "split": "unseen", "policy": r["run_label"], "makespan": r["unseen_mappo_makespan_mean"], "makespanSd": r["unseen_mappo_makespan_std"], "completeRate": r["completion_rate_unseen"], "soft": r["soft_unseen"], "hardExecuted": r["hard_executed_unseen"], "hardMask": r["hard_mask_unseen"], "travel": r["travel_unseen"], "setup": r["setup_unseen"], "move": r["move_unseen"]})
    else:
        for split in ["seen", "unseen"]:
            for pol in ["mappo", "nearest", "radiusPriority", "random"]:
                rows.append(norm_pytorch(pytorch_doc, split, pol))

    by_key = {(r["engine"], r["split"], r["policy"]): r for r in rows}
    browser_unseen_mappo = next((r for r in rows if r["engine"] == "browser-light" and r["split"] == "unseen" and r["policy"] == "MAPPO-trained"), None)
    pytorch_unseen = next((r for r in rows if r["engine"] == "pytorch-repeated-aggregate" and r["split"] == "unseen"), None) or next((r for r in rows if r["engine"] == "pytorch" and r["split"] == "unseen" and r["policy"] == "PyTorch MAPPO"), None)
    comparison = {
        "notes": [
            "Browser-light MAPPO는 브라우저 내 구조/시각화 검증용 경량 구현이다.",
            "PyTorch MAPPO는 분리된 연구용 학습 엔진이다.",
            "두 결과는 scenario generator를 정합화했더라도 학습 업데이트/후보 scoring/저장된 모델 시점이 다를 수 있어 직접 우열보다 역할 분리와 공정 조건 확인이 중요하다.",
        ],
        "browser_unseen_mappo_makespan": browser_unseen_mappo.get("makespan") if browser_unseen_mappo else None,
        "pytorch_unseen_mappo_makespan": pytorch_unseen.get("makespan") if pytorch_unseen else None,
        "pytorch_vs_browser_unseen_delta": (pytorch_unseen.get("makespan", 0) - browser_unseen_mappo.get("makespan", 0)) if browser_unseen_mappo and pytorch_unseen else None,
    }

    write_csv(outdir / "fair_comparison.csv", rows)
    (outdir / "fair_comparison.json").write_text(json.dumps({"rows": rows, "comparison": comparison}, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Browser-light MAPPO vs PyTorch MAPPO 공정 비교", "", *[f"- {n}" for n in comparison["notes"]], ""]
    if comparison["browser_unseen_mappo_makespan"] is not None:
        md += [
            f"- Browser-light unseen MAPPO makespan: {comparison['browser_unseen_mappo_makespan']:.2f}",
            f"- PyTorch unseen MAPPO makespan: {comparison['pytorch_unseen_mappo_makespan']:.2f}",
            f"- PyTorch - Browser delta: {comparison['pytorch_vs_browser_unseen_delta']:.2f}",
        ]
    (outdir / "fair_comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"rows": rows, "comparison": comparison, "outdir": str(outdir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
