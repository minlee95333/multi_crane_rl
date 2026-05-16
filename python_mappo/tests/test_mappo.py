"""Regression tests for MAPPO actor/critic shape and mask behaviour."""
from __future__ import annotations

import numpy as np
import torch

from python_mappo.env import CraneSchedulingEnv
from python_mappo.mappo import MAPPOAgent


def _make_agent(cfg):
    env = CraneSchedulingEnv(cfg)
    obs, masks, glob = env.reset(101)
    agent = MAPPOAgent(obs_dim=obs.shape[-1], state_dim=glob.shape[-1], cfg=cfg)
    return env, agent


def test_actor_respects_action_mask(cfg):
    """Logits at masked positions are sent to -1e9; their probability under
    Categorical(logits) must be effectively zero."""
    env, agent = _make_agent(cfg)
    obs, masks, _ = env.reset(101)
    # Force a partial mask: first candidate only.
    masked = masks.copy()
    masked[:, 1:] = False
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    mask_t = torch.as_tensor(masked, dtype=torch.bool)
    dist = agent.actor.distribution(obs_t, mask_t)
    probs = dist.probs.detach().numpy()
    assert np.allclose(probs[:, 1:], 0.0, atol=1e-6), (
        f"masked logits leaked into the distribution: {probs}"
    )
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5)


def test_greedy_act_picks_an_unmasked_slot(cfg):
    """`act_np(greedy=True)` must never select a masked slot."""
    env, agent = _make_agent(cfg)
    obs, masks, _ = env.reset(101)
    actions = agent.act_np(obs, masks, greedy=True)
    for ci, a in enumerate(actions):
        assert masks[ci, a], f"crane {ci} picked a masked slot {a}: mask={masks[ci]}"


def test_save_load_roundtrip_preserves_action(cfg, tmp_path):
    """Save and reload must reproduce the exact greedy action for a fixed obs."""
    env, agent = _make_agent(cfg)
    obs, masks, _ = env.reset(101)
    a_before = agent.act_np(obs, masks, greedy=True)
    path = tmp_path / "model.pt"
    agent.save(str(path))
    reloaded = MAPPOAgent.load(str(path))
    a_after = reloaded.act_np(obs, masks, greedy=True)
    assert list(a_before) == list(a_after)
