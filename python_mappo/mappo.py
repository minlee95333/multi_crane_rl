from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import numpy as np
import torch
import torch.nn.functional as F

from .networks import CandidateActor, CentralCritic


@dataclass
class Transition:
    agent: int
    obs: np.ndarray
    mask: np.ndarray
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class MAPPOAgent:
    def __init__(self, obs_dim: int, state_dim: int, cfg: Dict, device: str = 'cpu'):
        self.cfg = cfg
        self.device = torch.device(device)
        m = cfg.get('mappo', {})
        h = int(m.get('hidden_dim', 96))
        self.actor = CandidateActor(obs_dim, h).to(self.device)
        self.critic = CentralCritic(state_dim, h).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(m.get('actor_lr', 1e-3)))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(m.get('critic_lr', 2e-3)))
        self.stats = {'updates': 0, 'actorShareMode': 'typeShared', 'craneType': '50톤 모바일 크레인', 'actorDim': obs_dim, 'criticDim': state_dim}

    @torch.no_grad()
    def act(self, obs, mask, greedy=False, epsilon=0.0):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        dist = self.actor.distribution(obs_t, mask_t)
        if greedy:
            actions = torch.argmax(self.actor(obs_t, mask_t), dim=-1)
        else:
            actions = dist.sample()
            if epsilon > 0:
                for i in range(actions.shape[0]):
                    valid = torch.nonzero(mask_t[i], as_tuple=False).flatten().tolist()
                    if valid and random.random() < epsilon:
                        actions[i] = random.choice(valid)
        logp = dist.log_prob(actions)
        return actions.cpu().numpy().tolist(), logp.cpu().numpy().tolist()

    @torch.no_grad()
    def act_np(self, obs, mask, greedy=True):
        actions, _ = self.act(obs, mask, greedy=greedy, epsilon=0.0)
        return actions

    @torch.no_grad()
    def value(self, state):
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if st.ndim == 1: st = st.unsqueeze(0)
        return self.critic(st).cpu().numpy()

    def update(self, transitions: List[Transition]) -> Dict[str, float]:
        if not transitions:
            return {}
        m = self.cfg.get('mappo', {})
        gamma = float(m.get('gamma', 0.92)); lam = float(m.get('gae_lambda', 0.95))
        clip_eps = float(m.get('clip_eps', 0.2)); ent_coef=float(m.get('entropy_coef',0.01)); vf_coef=float(m.get('value_coef',0.5))
        max_grad=float(m.get('max_grad_norm',0.5)); epochs=int(m.get('update_epochs',4)); mb=int(m.get('minibatch_size',256))
        rewards=np.array([t.reward for t in transitions], dtype=np.float32)
        values=np.array([t.value for t in transitions], dtype=np.float32)
        dones=np.array([t.done for t in transitions], dtype=np.float32)
        adv=np.zeros_like(rewards)
        agents=np.array([t.agent for t in transitions], dtype=np.int32)
        for agent_id in sorted(set(agents.tolist())):
            idxs=np.where(agents==agent_id)[0]
            last=0.0
            for pos in reversed(range(len(idxs))):
                i=idxs[pos]
                nextv = 0.0 if pos == len(idxs)-1 or dones[i] else values[idxs[pos+1]]
                delta = rewards[i] + gamma*nextv*(1-dones[i]) - values[i]
                last = delta + gamma*lam*(1-dones[i])*last
                adv[i] = last
        ret = adv + values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs=torch.as_tensor(np.stack([t.obs for t in transitions]), dtype=torch.float32, device=self.device)
        mask=torch.as_tensor(np.stack([t.mask for t in transitions]), dtype=torch.bool, device=self.device)
        state=torch.as_tensor(np.stack([t.state for t in transitions]), dtype=torch.float32, device=self.device)
        action=torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        old_logp=torch.as_tensor([t.logp for t in transitions], dtype=torch.float32, device=self.device)
        adv_t=torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        ret_t=torch.as_tensor(ret, dtype=torch.float32, device=self.device)
        n=len(transitions); idx=np.arange(n)
        last_losses={}
        for _ in range(epochs):
            np.random.shuffle(idx)
            for start in range(0,n,mb):
                b=idx[start:start+mb]
                dist=self.actor.distribution(obs[b], mask[b])
                logp=dist.log_prob(action[b])
                ratio=torch.exp(logp-old_logp[b])
                pg1=ratio*adv_t[b]
                pg2=torch.clamp(ratio,1-clip_eps,1+clip_eps)*adv_t[b]
                actor_loss=-torch.min(pg1,pg2).mean()
                entropy=dist.entropy().mean()
                value=self.critic(state[b])
                critic_loss=F.mse_loss(value, ret_t[b])
                loss=actor_loss + vf_coef*critic_loss - ent_coef*entropy
                self.actor_opt.zero_grad(); self.critic_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad)
                self.actor_opt.step(); self.critic_opt.step()
                self.stats['updates'] += 1
                last_losses={'loss':float(loss.detach()), 'actorLoss':float(actor_loss.detach()), 'criticLoss':float(critic_loss.detach()), 'entropy':float(entropy.detach())}
        return last_losses

    def save(self, path: str, extra: Optional[Dict] = None) -> None:
        torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(), 'cfg': self.cfg, 'stats': self.stats, 'extra': extra or {}}, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'MAPPOAgent':
        # weights_only=True blocks arbitrary pickle execution when loading untrusted .pt files.
        ckpt=torch.load(path, map_location=device, weights_only=True)
        obs_dim=ckpt['stats']['actorDim']; state_dim=ckpt['stats']['criticDim']
        obj=cls(obs_dim, state_dim, ckpt['cfg'], device=device)
        obj.actor.load_state_dict(ckpt['actor']); obj.critic.load_state_dict(ckpt['critic']); obj.stats=ckpt.get('stats', obj.stats)
        return obj
