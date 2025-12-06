# Seed Reliability Check

## 사용법 요약
1. `cd src/analysis/reliability`
2. 시드 실험 실행  
   `python seed_reliability.py --episodes 20 --seeds 0 1 2 42 999`
3. 시각화 생성  
   `python visualize_reliability.py --results results.json --output seed_reliability.png`
4. 결과 확인  
   - `results.json`: 시드별 승/무/패, 평균 승률, 표준편차, 95% CI  
   - `seed_reliability.png`: 시드 곡선 + 평균 승률 막대(에러바)

---

## 1. 시드별 실험 (seed_reliability.py)

```
python seed_reliability.py --episodes 20 --seeds 0 1 2 42 999
```

- `submission_files/` 안 주요 에이전트 다섯 개( Greedy, N-step, MTD(f), Double DQN, AlphaZero )를 모두 로드해 각 시드마다 20판씩 Kaggle `random`과 대전합니다.
- 결과는 `results.json`으로 저장되며 시드별 승/무/패 기록과 평균 승률, 표준편차, 95% 신뢰구간이 포함됩니다.

## 2. 시각화 (visualize_reliability.py)

```
python visualize_reliability.py --results results.json --output seed_reliability.png
```

- 왼쪽 그래프: 시드별 승률 궤적 → 특정 시드에서 급격히 떨어지는지 확인 가능.
- 오른쪽 그래프: 평균 승률에 95% 신뢰구간(에러바)를 붙여 안정성과 성능을 동시에 비교.
- `seed_reliability.png`는 보고서에 바로 넣을 수 있도록 200 DPI로 저장됩니다.

## 해석 포인트
- 곡선이 평평할수록, 에러바가 짧을수록 시드 민감도가 낮습니다.
- Greedy/MTD(f)/AlphaZero는 모든 시드에서 승률 1.0으로 완전히 안정적입니다.
- Double DQN은 평균 승률은 높지만 시드 1에서 급락해 표준편차·CI가 크며, 탐험률/탐색 깊이 보정 필요성을 시사합니다.

이 워크플로로 “시드가 바뀌어도 리그 결과를 믿을 수 있는가?”라는 질문에 데이터를 근거로 답할 수 있습니다.
