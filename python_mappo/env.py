from __future__ import annotations

from dataclasses import dataclass, asdict
import math, random
from typing import Dict, List, Tuple, Optional

import numpy as np


def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


@dataclass
class Crane:
    id: str
    x: float
    y: float
    setup_x: float
    setup_y: float
    available: float = 0.0
    jobs: int = 0


@dataclass
class Lift:
    id: str
    x: float
    y: float
    done: bool = False
    assigned: Optional[str] = None


class CraneSchedulingEnv:
    """Stage-1 crane scheduling env matching the browser prototype conventions.

    - one crane = one agent
    - action = candidate slot, not raw lift ID
    - actual lifting radius is crane/setup-centered: distance(setup point, lift)
    - actual-radius hard overlap is infeasible/masked/resolved, soft buffer is reward shaping
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.nC = int(cfg.get('num_cranes', 3))
        self.nL = int(cfg.get('num_lifts', 24))
        self.K = int(cfg.get('candidate_k', 5))
        self.fixed_duration = float(cfg.get('fixed_duration', 25.0))
        self.setup_time = float(cfg.get('setup_time', 10.0))
        self.crane_radius = float(cfg.get('crane_radius', 18.0))
        self.max_steps = int(cfg.get('max_steps', 220))
        r = cfg.get('reward', {})
        self.r_single = float(r.get('r_single', 10.0))
        self.r_all = float(r.get('r_all', 100.0))
        self.r_same = float(r.get('r_same', 3.0))
        self.p_idle = float(r.get('p_idle', -0.5))
        self.p_inter_soft = float(r.get('p_inter_soft', -3.0))
        self.p_time = float(r.get('p_time', -0.1))
        self.p_move = float(r.get('p_move', -0.02))
        self.soft_buffer = float(r.get('soft_buffer', 6.0))
        self._lcg_state = 1
        self.reset(0)

    def _set_seed(self, seed: int):
        """Match the browser dashboard's makeRng(seed) LCG.

        JS uses uint32 arithmetic and rand(a,b)=a+rng()*(b-a). Keeping this
        generator here makes Python scenario seeds geometrically comparable with
        the browser dashboard.
        """
        self._lcg_state = int(seed or 1) & 0xFFFFFFFF

    def _rng(self) -> float:
        self._lcg_state = (1664525 * self._lcg_state + 1013904223) & 0xFFFFFFFF
        return self._lcg_state / 4294967296.0

    def _rand(self, a: float, b: float) -> float:
        return a + self._rng() * (b - a)

    def reset(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._set_seed(seed)
        self.hard_mask_total = 0
        self.executed_hard_total = 0
        self.step_count = 0
        self.events: List[Dict] = []
        # Match browser generateScenario(): cranes and lifts are uniform in the
        # 0-100 site coordinate frame, inside [8, 92].
        self.cranes = []
        for i in range(self.nC):
            x, y = self._rand(8.0, 92.0), self._rand(8.0, 92.0)
            self.cranes.append(Crane(f'C{i+1}', x, y, x, y))
        self.lifts = [Lift(f'L{i+1}', self._rand(8.0, 92.0), self._rand(8.0, 92.0)) for i in range(self.nL)]
        return self.observe()

    def done_count(self):
        return sum(1 for l in self.lifts if l.done)

    def is_done(self):
        return self.done_count() >= self.nL or self.step_count >= self.max_steps

    def setup_target(self, crane: Crane, lift: Lift) -> Tuple[float, float, bool]:
        cur = (crane.setup_x, crane.setup_y)
        lp = (lift.x, lift.y)
        d = dist(cur, lp)
        if d <= self.crane_radius + 1e-9:
            return crane.setup_x, crane.setup_y, True
        # Move along current-setup -> lift line, stopping crane_radius away from lift.
        ratio = (d - self.crane_radius) / max(d, 1e-9)
        nx = crane.setup_x + (lift.x - crane.setup_x) * ratio
        ny = crane.setup_y + (lift.y - crane.setup_y) * ratio
        return nx, ny, False

    def candidate_outcome(self, ci: int, li: int) -> Dict:
        c, l = self.cranes[ci], self.lifts[li]
        sx, sy, same = self.setup_target(c, l)
        move = dist((c.setup_x, c.setup_y), (sx, sy))
        travel = move / 10.0
        setup = 0.0 if same else self.setup_time
        start = c.available
        finish = start + travel + setup + self.fixed_duration
        actual = dist((sx, sy), (l.x, l.y))
        return dict(ci=ci, li=li, sx=sx, sy=sy, same=same, move=move, travel=travel, setup=setup, start=start, finish=finish, actual=actual)

    def _overlaps(self, a_start, a_finish, b_start, b_finish):
        return a_start < b_finish and b_start < a_finish

    def risk_counts(self, out: Dict, planned: Optional[List[Dict]] = None) -> Tuple[int, int]:
        hard = soft = 0
        planned = planned or []
        for e in self.events + planned:
            if not self._overlaps(out['start'], out['finish'], e['start'], e['finish']):
                continue
            cd = dist((out['sx'], out['sy']), (e['radiusCenterX'], e['radiusCenterY']))
            hard_lim = out['actual'] + e['actualLiftRadius']
            if cd < hard_lim - 1e-9:
                hard += 1
            elif cd < hard_lim + self.soft_buffer - 1e-9:
                soft += 1
        return hard, soft

    def candidate_actions(self, ci: int, reserved: Optional[set] = None) -> List[Optional[int]]:
        reserved = reserved or set()
        avail = [i for i, l in enumerate(self.lifts) if not l.done and i not in reserved]
        if not avail:
            return [None] * self.K
        feasible=[]; blocked=[]
        for li in avail:
            out = self.candidate_outcome(ci, li)
            hard, soft = self.risk_counts(out)
            # Candidate-K composition: same-radius continuity, earliest finish, nearest/move, low risk.
            score = (0 if out['same'] else 1000) + out['finish'] + 0.18*out['move'] + 4*soft
            (blocked if hard>0 else feasible).append((score, li))
        scored = feasible if feasible else blocked
        scored.sort(key=lambda x: x[0])
        cands = [li for _, li in scored[:self.K]]
        return cands + [None] * (self.K - len(cands))

    def action_masks(self):
        masks=[]
        for ci in range(self.nC):
            cands=self.candidate_actions(ci)
            masks.append([li is not None for li in cands])
        return np.array(masks, dtype=np.bool_)

    def _candidate_features(self, ci: int, li: Optional[int]):
        c = self.cranes[ci]
        if li is None:
            return [0.0]*12
        l = self.lifts[li]
        out = self.candidate_outcome(ci, li)
        hard, soft = self.risk_counts(out)
        rem = max(1, self.nL - self.done_count())
        nearby = sum(1 for x in self.lifts if not x.done and dist((x.x,x.y),(l.x,l.y)) <= self.crane_radius)
        return [
            1.0,
            float(out['same']),
            out['move']/100.0,
            out['travel']/20.0,
            out['setup']/max(1.0,self.setup_time),
            out['finish']/500.0,
            out['actual']/max(1.0,self.crane_radius),
            hard/5.0,
            soft/5.0,
            nearby/rem,
            l.x/100.0,
            l.y/100.0,
        ]

    def observe(self):
        obs=[]
        masks=[]
        for ci,c in enumerate(self.cranes):
            local=[c.setup_x/100.0,c.setup_y/100.0,c.available/500.0,c.jobs/max(1,self.nL),self.done_count()/max(1,self.nL)]
            cands=self.candidate_actions(ci)
            masks.append([li is not None for li in cands])
            cand_feats=[local+self._candidate_features(ci,li) for li in cands]
            obs.append(cand_feats)
        return np.array(obs,dtype=np.float32), np.array(masks,dtype=np.bool_), self.global_state()

    def global_state(self):
        av=np.array([c.available for c in self.cranes], dtype=np.float32)
        return np.array([
            self.done_count()/max(1,self.nL),
            (self.nL-self.done_count())/max(1,self.nL),
            av.min()/500.0 if len(av) else 0,
            av.max()/500.0 if len(av) else 0,
            av.mean()/500.0 if len(av) else 0,
            av.std()/500.0 if len(av) else 0,
            self.nC/10.0,
            self.nL/100.0,
            self.K/12.0,
            self.step_count/max(1,self.max_steps),
            sum(e.get('softConflict',0) for e in self.events)/max(1,self.nL),
            sum(e.get('hardMask',0) for e in self.events)/max(1,self.nL),
            sum(e.get('move',0) for e in self.events)/1000.0,
            sum(e.get('setup',0) for e in self.events)/1000.0,
        ], dtype=np.float32)

    def step(self, actions: List[int]):
        self.step_count += 1
        rewards = np.zeros(self.nC, dtype=np.float32)
        planned=[]; used=set(); hard_masks=0; soft_total=0
        order = sorted(range(self.nC), key=lambda i: self.cranes[i].available)
        for ci in order:
            cands = self.candidate_actions(ci, used)
            ai = int(actions[ci]) if ci < len(actions) else 0
            li = cands[ai] if 0 <= ai < len(cands) else None
            if li is None:
                if self.done_count() < self.nL:
                    rewards[ci] += self.p_idle
                continue
            out = self.candidate_outcome(ci, li)
            hard, soft = self.risk_counts(out, planned)
            if hard > 0:  # hard overlap is infeasible; try the next feasible candidate before idling.
                hard_masks += hard
                self.hard_mask_total += hard
                replacement = None
                for alt_li in cands:
                    if alt_li is None or alt_li == li or alt_li in used:
                        continue
                    alt = self.candidate_outcome(ci, alt_li)
                    alt_hard, alt_soft = self.risk_counts(alt, planned)
                    if alt_hard == 0:
                        replacement = (alt_li, alt, alt_soft)
                        break
                if replacement is None:
                    rewards[ci] += self.p_idle
                    continue
                li, out, soft = replacement
                hard = 0
            c, l = self.cranes[ci], self.lifts[li]
            event = {
                'craneId': c.id, 'craneIndex': ci, 'liftId': l.id, 'liftIndex': li,
                'start': out['start'], 'finish': out['finish'], 'travel': out['travel'], 'setup': out['setup'], 'duration': self.fixed_duration,
                'fromX': c.setup_x, 'fromY': c.setup_y, 'toX': out['sx'], 'toY': out['sy'],
                'radiusCenterX': out['sx'], 'radiusCenterY': out['sy'], 'liftX': l.x, 'liftY': l.y,
                'actualLiftRadius': out['actual'], 'dangerRadius': out['actual'] + self.soft_buffer,
                'sameRadius': out['same'], 'softConflict': soft, 'hardMask': hard, 'move': out['move']
            }
            planned.append(event); used.add(li); soft_total += soft
            task_time = out['finish'] - out['start']
            rewards[ci] += self.r_single + (self.r_same if out['same'] else 0.0) + self.p_time*task_time + self.p_move*out['move'] + self.p_inter_soft*soft
            c.setup_x, c.setup_y, c.available, c.jobs = out['sx'], out['sy'], out['finish'], c.jobs+1
            l.done, l.assigned = True, c.id
        self.events.extend(planned)
        done = self.is_done()
        if self.done_count() >= self.nL:
            rewards += self.r_all / max(1,self.nC)
        obs, masks, glob = self.observe()
        info = self.summary()
        info['stepHardMask'] = hard_masks
        info['stepSoft'] = soft_total
        return obs, masks, glob, rewards, done, info

    def summary(self):
        makespan = max([c.available for c in self.cranes] + [0.0])
        return {
            'done': self.done_count(), 'total': self.nL, 'makespan': round(makespan, 3),
            'reward': 0.0, 'softInter': int(sum(e.get('softConflict',0) for e in self.events)),
            # Split actual executed hard overlaps from blocked/masked candidates.
            'hardExecuted': int(self.executed_hard_total),
            'hardMask': int(self.hard_mask_total),
            'hardInter': int(self.hard_mask_total),  # backward-compatible alias
            'travelTotal': sum(e.get('travel',0.0) for e in self.events),
            'setupTotal': sum(e.get('setup',0.0) for e in self.events),
            'moveTotal': sum(e.get('move',0.0) for e in self.events),
            'actualLiftRadiusAvg': float(np.mean([e['actualLiftRadius'] for e in self.events])) if self.events else 0.0,
            'actualDangerRadiusAvg': float(np.mean([e['dangerRadius'] for e in self.events])) if self.events else 0.0,
            'events': list(self.events),
            'cranes': [asdict(c) for c in self.cranes],
        }

    def run_policy(self, policy: str, model=None, seed: int = 0, greedy=True):
        self.reset(seed)
        total_reward=0.0
        while not self.is_done():
            obs,masks,glob = self.observe()
            if policy == 'random':
                actions=[self.rng.choice([i for i,m in enumerate(masks[ci]) if m] or [0]) for ci in range(self.nC)]
            elif policy in ('nearest','radiusPriority'):
                actions=[0 for _ in range(self.nC)]
            elif policy == 'mappo' and model is not None:
                actions=model.act_np(obs,masks,greedy=greedy)
            else:
                actions=[self.rng.choice([i for i,m in enumerate(masks[ci]) if m] or [0]) for ci in range(self.nC)]
            *_ , rewards, done, info = self.step(actions)
            total_reward += float(np.sum(rewards))
        s=self.summary(); s['reward']=total_reward
        return s
