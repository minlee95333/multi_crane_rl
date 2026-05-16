"""Regression tests for CraneSchedulingEnv invariants."""
from __future__ import annotations

import numpy as np

from python_mappo.env import CraneSchedulingEnv


def test_scenario_generator_is_deterministic(cfg):
    """The LCG seed must produce byte-identical crane/lift placements across
    runs so browser-light and PyTorch results stay seed-comparable."""
    env_a = CraneSchedulingEnv(cfg)
    env_b = CraneSchedulingEnv(cfg)
    env_a.reset(101)
    env_b.reset(101)
    crane_a = [(c.x, c.y) for c in env_a.cranes]
    crane_b = [(c.x, c.y) for c in env_b.cranes]
    lift_a = [(l.x, l.y) for l in env_a.lifts]
    lift_b = [(l.x, l.y) for l in env_b.lifts]
    assert crane_a == crane_b, "crane placement diverged for the same seed"
    assert lift_a == lift_b, "lift placement diverged for the same seed"


def test_hard_overlap_is_never_executed(cfg):
    """Hard lifting-radius overlap must be blocked before event recording.
    `hard_mask_total` may grow, but no recorded event should carry `hardMask>0`.
    """
    env = CraneSchedulingEnv(cfg)
    for seed in range(101, 111):
        result = env.run_policy("nearest", seed=seed)
        for ev in result["events"]:
            assert ev.get("hardMask", 0) == 0, (
                f"seed {seed} recorded an event with hard overlap: {ev}"
            )


def test_baseline_policies_complete_all_lifts(cfg):
    """Stage-1 should be fully solvable by the simple baselines on the train
    seed range; otherwise the env has regressed (e.g. wrong idle path)."""
    env = CraneSchedulingEnv(cfg)
    for policy in ("nearest", "radiusPriority", "random"):
        result = env.run_policy(policy, seed=101)
        assert result["done"] == result["total"], (
            f"policy={policy} failed to complete all lifts: {result['done']}/{result['total']}"
        )


def test_nearest_and_radius_priority_can_diverge(cfg):
    """After Round 2 they must be distinct policies — at least one seed should
    produce a different makespan."""
    env = CraneSchedulingEnv(cfg)
    diffs = 0
    for seed in range(101, 116):
        n = env.run_policy("nearest", seed=seed)
        rp = env.run_policy("radiusPriority", seed=seed)
        if (n["makespan"], n["moveTotal"]) != (rp["makespan"], rp["moveTotal"]):
            diffs += 1
    assert diffs > 0, "nearest and radiusPriority are still mathematically identical"


def test_summary_does_not_emit_hard_executed(cfg):
    """Round 1 removed the always-zero hardExecuted field from env summary."""
    env = CraneSchedulingEnv(cfg)
    result = env.run_policy("nearest", seed=101)
    assert "hardExecuted" not in result
    assert "hardMask" in result
