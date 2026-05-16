# Curriculum Stabilization Summary

## L1_2C12L (2C/12L)
- episodes: 90
- seen: MAPPO 229.28, Nearest 224.80, 개선율 -1.99%
- unseen: MAPPO 239.16, Nearest 242.68, 개선율 1.45%
- completion: seen 100.0%, unseen 100.0%
- hardExecuted: seen 0.0, unseen 0.0
- best checkpoint: ep 36, seen makespan 229.28

## L2_3C24L (3C/24L)
- episodes: 150
- seen: MAPPO 319.10, Nearest 347.84, 개선율 8.26%
- unseen: MAPPO 349.27, Nearest 332.22, 개선율 -5.13%
- completion: seen 100.0%, unseen 100.0%
- hardExecuted: seen 0.0, unseen 0.0
- best checkpoint: ep 90, seen makespan 319.10

## L3_4C36L (4C/36L)
- episodes: 210
- seen: MAPPO 392.66, Nearest 401.11, 개선율 2.11%
- unseen: MAPPO 371.56, Nearest 392.18, 개선율 5.26%
- completion: seen 100.0%, unseen 100.0%
- hardExecuted: seen 0.0, unseen 0.0
- best checkpoint: ep 168, seen makespan 392.66

Final model: python_mappo/outputs_curriculum_stable/curriculum_mappo_model.pt
PPO updates: 1176
Elapsed seconds: 142.2