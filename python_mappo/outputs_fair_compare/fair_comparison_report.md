# Browser-light MAPPO vs PyTorch MAPPO 공정 비교

- Browser-light MAPPO는 브라우저 내 구조/시각화 검증용 경량 구현이다.
- PyTorch MAPPO는 분리된 연구용 학습 엔진이다.
- 두 결과는 scenario generator를 정합화했더라도 학습 업데이트/후보 scoring/저장된 모델 시점이 다를 수 있어 직접 우열보다 역할 분리와 공정 조건 확인이 중요하다.

- Browser-light unseen MAPPO makespan: 297.33
- PyTorch unseen MAPPO makespan: 332.46
- PyTorch - Browser delta: 35.12