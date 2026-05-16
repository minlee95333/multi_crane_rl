# Multi-Crane RL Scheduling

**다중 모바일 크레인(50톤) 양중 스케줄링**을 다중 에이전트 강화학습(MAPPO)으로 푸는 연구·프로토타입 프로젝트.

크레인 1대를 1개의 에이전트로 보고, 여러 양중물(lifts)을 가장 짧은 makespan으로 안전하게 끝내는 정책을 학습합니다. 동일한 환경을 두 갈래로 제공합니다.

- **브라우저 대시보드** (`index.html`) — 즉시 시각화·실험·계획 생성용. Q-learning baseline과 경량 MAPPO를 포함.
- **PyTorch MAPPO 엔진** (`python_mappo/`) — 연구용 본격 학습·반복 검증·sweep·커리큘럼.

두 엔진은 동일한 시나리오 생성기(LCG)를 사용해 seed 단위로 직접 비교할 수 있습니다.

---

## 폴더 구조

```
multi_crane_rl/
├─ app.py                  # 대시보드 정적 HTTP 서버 (포트 8000)
├─ index.html              # 대시보드 UI + 브라우저-light 학습/평가 코드
├─ deploy_*.py, run_deploy.sh, tunnel.js
│                          # 외부 공개용 터널 배포 (localtunnel / ssh -R)
├─ package.json            # npm scripts (학습/평가/실험 진입점)
│
├─ curriculum_runs/
│   ├─ crane_script.js     # 대시보드의 핵심 로직(Node vm에서도 실행 가능)
│   ├─ run_stage.js / run_eval.js
│   │                      # 헤드리스 커리큘럼 학습 + seen/unseen 평가
│   ├─ current_rl_validation.js / current_rl_validation_summary.py
│   │                      # before/after RL + seen/unseen 비교 한 번에
│   ├─ smoke_async.js, profile_episode.js, measure_episode.js, inspect_final_event.js
│   ├─ model_stage*.json   # 단계별 학습된 weight
│   └─ eval_*.json / summary_*.json
│
└─ python_mappo/
    ├─ env.py              # CraneSchedulingEnv (Gym-like)
    ├─ networks.py         # CandidateActor + CentralCritic
    ├─ mappo.py            # MAPPO (rollout, GAE-λ, PPO clipped update)
    ├─ train.py            # 학습 + 평가 + best-checkpoint 보존
    ├─ evaluate.py         # 저장된 모델로 평가만 실행
    ├─ curriculum.py       # 2C/12L → 3C/24L → 4C/36L 커리큘럼
    ├─ experiments.py      # episode 수를 바꿔 반복 학습
    ├─ repeated_validation.py
    │                      # seed group 3개로 반복 검증
    ├─ reward_sweep.py     # reward 계수 sweep
    ├─ fair_compare.py     # browser-light vs PyTorch 정합 비교
    ├─ config.yaml         # 환경·보상·MAPPO 하이퍼파라미터
    ├─ outputs*/           # 학습 결과(.pt, .json, .csv)
    └─ archive/            # 옛 실험 결과 보관
```

---

## 빠른 시작

### 브라우저 대시보드

```bash
python app.py
# → http://localhost:8000/index.html
```

좌측 패널에서 "학습 시작" → "정책 평가" / "Baseline 비교" / "Curriculum 1→2→3 학습". 2D 시뮬레이션·간트·리플레이가 함께 동작합니다. 학습된 모델은 "모델 저장 JSON"으로 export.

### PyTorch MAPPO 학습/평가

```bash
# 단일 학습 (150 episodes ≈ 10초)
npm run train:pytorch-mappo
# 또는
python -m python_mappo.train --episodes 150 --outdir python_mappo/outputs

# 저장된 모델로 평가만
npm run eval:pytorch-mappo

# 커리큘럼 학습 (3단계)
npm run curriculum:pytorch-mappo       # [60, 90, 120] ep
npm run curriculum:stabilize           # [90, 150, 210] ep

# 3그룹 반복 검증 (보고서용)
npm run repeat:pytorch-mappo

# Reward 계수 sweep
npm run sweep:pytorch-reward

# 브라우저 vs PyTorch 공정 비교 리포트
npm run compare:mappo
```

### 헤드리스 브라우저 검증

```bash
node curriculum_runs/current_rl_validation.js 150
# → curriculum_runs/current_rl_validation_result.json
```

### 회귀 테스트

```bash
pytest python_mappo/tests
```

---

## 핵심 설계

- **Action = Candidate-K 슬롯**: raw lift ID가 아닌 K=5개 후보 슬롯 인덱스. lift 수 M이 변해도 actor 차원이 동일 → curriculum learning 가능.
- **Hard 안전제약 = action mask** / **Soft 안전제약 = reward shaping**. Hard 양중반경 overlap은 실행 자체가 차단되며 `hardMask` 카운터로만 보고됨 (`hardExecuted`는 mask 정상 작동 시 항상 0).
- **Type-shared actor**: "50톤 모바일 크레인" 한 종류이므로 모든 에이전트가 단일 actor weight 공유 — 샘플 효율 ↑.
- **Best-checkpoint 보존**: 학습 중 주기적으로 seen-seed evaluate → makespan 최저 시점의 weight를 따로 저장. 마지막 weight보다 더 안정적.
- **두 엔진 시나리오 정합**: 동일 LCG로 seed 단위 직접 비교 (`fair_compare.py`).

### MAPPO 하이퍼파라미터 (`python_mappo/config.yaml`)

| 항목 | 값 | 비고 |
|---|---|---|
| `num_cranes` / `num_lifts` | 3 / 24 | Stage-1 표준 |
| `fixed_duration` / `setup_time` | 25 / 10 분 | 양중 시간 / 재설치 시간 |
| `crane_radius` / `crane_speed` | 18 / 10 | 단일 작업 반경 / 이동속도 |
| `candidate_k` | 5 | top-K 후보 슬롯 |
| `gamma` / `gae_lambda` / `clip_eps` | 0.92 / 0.95 / 0.20 | PPO |
| `actor_lr` / `critic_lr` | 1e-3 / 2e-3 | |
| `hidden_dim` | 96 | 양 네트워크 |
| `eps_start` / `eps_end` | 0.35 / 0.05 | ε-greedy 선형 감쇠 |

---

## 결과물 위치

| 산출물 | 경로 |
|---|---|
| 학습 모델 | `python_mappo/outputs/pytorch_mappo_model.pt` |
| 학습 검증 결과(전체 메타+seen/unseen+대표 episode) | `python_mappo/outputs/pytorch_mappo_validation_result.json` |
| Learning curve CSV | `python_mappo/outputs/learning_curve.csv` |
| 커리큘럼 단계별 best 모델 | `python_mappo/outputs_curriculum_stable/L{1,2,3}_*_best_model.pt` |
| 반복 검증 요약 | `python_mappo/outputs_repeated/report_summary.md` |
| Browser vs PyTorch 비교 | `python_mappo/outputs_fair_compare/fair_comparison_report.md` |
| 브라우저-light 학습 결과 | `curriculum_runs/current_rl_validation_result.json` |

---

## 메트릭 해석 순서

1. **makespan** — 가장 중요. 전체 양중 종료 시각.
2. **완료율 + HardMask 정상 작동** — 모든 lift가 끝났는가, hard overlap이 0회 실행됐는가.
3. **travel / setup / move** 효율 — 운영 비용 직결.
4. **soft 노출** — 허용 가능한 안전 반경 근접 작업. 실패가 아니라 보조 지표.
5. **reward** — 학습 신호일 뿐 절대 비교 기준이 아님.

---

## 외부 공개

```bash
# 어디서나 (localtunnel)
python deploy_temp.py

# SSH reverse tunnel
python deploy_ssh.py

# Linux 전용 (재기동 루프)
bash run_deploy.sh
```

URL이 `tempfile.gettempdir()/crane_public_url.txt`에 기록됩니다. 인증이 없으므로 학습 모델과 import 흐름을 공개하기 전에 주의하세요.

---

## 환경 요구

- Python 3.10+, PyTorch 2.x, PyYAML, NumPy
- Node.js 18+ (브라우저 헤드리스 러너 및 localtunnel용)
- Windows / macOS / Linux 모두 지원
