# Stage-1 PyTorch MAPPO 반복 검증 요약

- 반복 run 수: 3
- Unseen MAPPO makespan 평균: 327.33
- Unseen MAPPO makespan run 간 표준편차: 12.73
- Unseen Nearest 대비 평균 개선율: 1.58%
- Seen Nearest 대비 평균 개선율: 6.37%
- Unseen 완료율 평균: 100.00%
- Unseen 실행 hard overlap 평균: 0.00
- Unseen HardMask 평균: 15.31

## Run별 결과

### group1_train101-110_unseen201-230
- train seeds: 101~110
- episodes: 120
- seen MAPPO makespan: 313.65 / Nearest 개선율 9.83%
- unseen MAPPO makespan: 342.88 / Nearest 개선율 1.49%
- hardExecuted unseen: 0.00, HardMask unseen: 15.93

### group2_train111-120_unseen231-260
- train seeds: 111~120
- episodes: 120
- seen MAPPO makespan: 337.25 / Nearest 개선율 1.23%
- unseen MAPPO makespan: 327.40 / Nearest 개선율 -1.84%
- hardExecuted unseen: 0.00, HardMask unseen: 13.73

### group3_train121-130_unseen261-290
- train seeds: 121~130
- episodes: 120
- seen MAPPO makespan: 298.79 / Nearest 개선율 8.04%
- unseen MAPPO makespan: 311.70 / Nearest 개선율 5.09%
- hardExecuted unseen: 0.00, HardMask unseen: 16.27
