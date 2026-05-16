# PyTorch MAPPO engine

별도 학습 엔진입니다. 기존 `index.html` 브라우저 대시보드는 시각화/비교 기준선으로 유지합니다.

## 실행

프로젝트 루트(`multi_crane_rl/`)에서 실행하세요:

```bash
python -m python_mappo.train --episodes 150 --outdir python_mappo/outputs
```

결과:

- `python_mappo/outputs/pytorch_mappo_model.pt`
- `python_mappo/outputs/pytorch_mappo_validation_result.json`

## 평가만 실행

```bash
python3 -m python_mappo.evaluate \
  --model python_mappo/outputs/pytorch_mappo_model.pt \
  --seed-start 201 --seed-count 30
```

## 현재 범위

- Stage-1 3 cranes / 24 lifts
- action = Candidate-K slot
- type-shared actor
- centralized critic
- actual lifting radius hard overlap은 실행 불가 처리
- soft exposure는 reward shaping
- deterministic greedy evaluation
