# Stage-1 PyTorch MAPPO 반복 검증 요약

- 반복 run 수: 3
- Unseen MAPPO makespan 평균: 332.46
- Unseen MAPPO makespan run 간 표준편차: 7.04
- Unseen Nearest 대비 평균 개선율: -0.23%
- Seen Nearest 대비 평균 개선율: 12.13%
- Unseen 완료율 평균: 100.00%
- Unseen 실행 hard overlap 평균: 0.00
- Unseen HardMask 평균: 16.89

## Run별 결과

### group1_train101-110_unseen201-230
- train seeds: 101~110
- episodes: 120
- seen MAPPO makespan: 312.62 / Nearest 개선율 11.45%
- unseen MAPPO makespan: 341.85 / Nearest 개선율 -0.59%
- hardExecuted unseen: 0.00, HardMask unseen: 16.87

### group2_train111-120_unseen231-260
- train seeds: 111~120
- episodes: 120
- seen MAPPO makespan: 325.60 / Nearest 개선율 11.27%
- unseen MAPPO makespan: 324.91 / Nearest 개선율 0.91%
- hardExecuted unseen: 0.00, HardMask unseen: 16.80

### group3_train121-130_unseen261-290
- train seeds: 121~130
- episodes: 120
- seen MAPPO makespan: 279.00 / Nearest 개선율 13.67%
- unseen MAPPO makespan: 330.60 / Nearest 개선율 -1.01%
- hardExecuted unseen: 0.00, HardMask unseen: 17.00
